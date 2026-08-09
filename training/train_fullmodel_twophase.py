"""
Two-Phase FullModel Training (48M params)
==========================================
Phase 1: Train on synthetic data (Field II or kWave hyperechoic only), runs ONCE.
Phase 2: Fine-tune on clinical data at multiple train_caps.

Usage:
    python train_fullmodel_twophase.py --sim_type fieldii --train_caps 200,300,400,500
    python train_fullmodel_twophase.py --sim_type kwave --train_caps 200,300,400,500
"""

import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import gaussian; from skimage.transform import resize as sk_resize
from sewar import vifp
from model.fullmodel import FullModel
from utils.clinical_dataloading import ClinicalLabelledDataset
from utils.balancedLoss import BalancedLoss
from utils.perceptual_loss import PerceptualLoss
from utils.otherUtils import compute_ncc
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--sim_type", type=str, required=True, choices=["fieldii","kwave"])
parser.add_argument("--train_caps", type=str, default="200,300,400,500")
parser.add_argument("--p1_epochs", type=int, default=80)
parser.add_argument("--p2_epochs", type=int, default=50)
args = parser.parse_args()
TRAIN_CAPS = [int(x) for x in args.train_caps.split(",")]; SIM_TYPE = args.sim_type

if SIM_TYPE == "fieldii":
    synthetic_images = "data/FieldData/tumor_images_final.npy"
    synthetic_labels = "data/FieldData/tumor_labels_elastography_image.npy"
    NUM_ECHO_TYPES = 1
elif SIM_TYPE == "kwave":
    synthetic_images = "data/train/tumor_images_final.npy"
    synthetic_labels = "data/train/tumor_labels_elastography_image.npy"
    NUM_ECHO_TYPES = 3

clinical_rf_folder = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"
clinical_label_folder = "data/RivazElastographyLabels"
clinical_mask_folder = "data/RFLigamentMasks"
PHASE1_EPOCHS=args.p1_epochs; PHASE1_START_LR=3e-4; PHASE1_MAX_LR=1e-3; PHASE1_BATCH_SIZE=4; PHASE1_PATIENCE=50
PHASE2_EPOCHS=args.p2_epochs; PHASE2_LR=3e-5; PHASE2_BATCH_SIZE=4
USE_OHEM=True; OHEM_WARMUP=5; OHEM_TOP_K=0.7; USE_MIXUP=True; MIXUP_ALPHA=0.2; T_0=10; T_MULT=1
WEIGHT_L1=1.0; WEIGHT_SSIM=0.5; WEIGHT_GDL=0.5; WEIGHT_TV=0.002; WEIGHT_MEAN=0.2
WEIGHT_CONTRAST=0.3; WEIGHT_RANGE=0.1; WEIGHT_PERCEPTUAL=0.1
CROP_TOP=40; CROP_BOTTOM=170; BLUR_SIGMA=2; SPECKLE_STR=0.35; SPECKLE_COR=0.8
CLINICAL_VAL_COUNT=100; CLINICAL_TEST_COUNT=100
WANDB_PROJECT="elastography"; P1_SAVE_DIR=f"model/best_fullmodel_{SIM_TYPE}"

def synth_to_clinical(label, seed=None):
    rng=np.random.RandomState(seed); lo,hi=np.percentile(label,5),np.percentile(label,95)
    label=np.clip(label,lo,hi); label=(label-lo)/(hi-lo+1e-8)
    c=label[CROP_TOP:CROP_BOTTOM,:]; r=sk_resize(c,(220,200),order=1,preserve_range=True).astype(np.float32)
    b=gaussian(r,sigma=BLUR_SIGMA); n=gaussian(rng.randn(*b.shape).astype(np.float32),sigma=SPECKLE_COR)
    return np.clip(b*(1+SPECKLE_STR*n),0,1).astype(np.float32)

class TransformedSyntheticDataset(Dataset):
    def __init__(s,ip,lp,augment=False,transform_labels=True):
        super().__init__(); s.images=np.load(ip,mmap_mode='r'); s.labels=np.load(lp,mmap_mode='r'); s.augment=augment; s.transform_labels=transform_labels
    def __len__(s): return len(s.images)
    def __getitem__(s,idx):
        img=s.images[idx].copy(); lbl=np.clip(s.labels[idx].copy().astype(np.float32)/255.0,0,1)
        if s.transform_labels: lbl=synth_to_clinical(lbl,seed=idx)
        if s.augment:
            if np.random.rand()<0.5: img=img[:,::-1,:].copy(); lbl=lbl[:,::-1].copy()
            if np.random.rand()<0.5: img=img+np.random.randn(*img.shape).astype(np.float32)*np.random.uniform(0.001,0.02)*np.abs(img).max()
            if np.random.rand()<0.5: img=img*np.random.uniform(0.85,1.15)
        img=torch.from_numpy(img.copy()).permute(2,0,1).float(); img=img/(img.abs().max()+1e-8)
        return img, torch.from_numpy(lbl.copy()).unsqueeze(0).float().clamp(0.0,1.0)

class ContrastAwareLoss(nn.Module):
    def __init__(s,wc=0.3,wr=0.1,ps=16): super().__init__(); s.wc=wc; s.wr=wr; s.ps=ps
    def lv(s,x,ps):
        p=ps//2; lm=F.avg_pool2d(x,ps,stride=1,padding=p); lm2=F.avg_pool2d(x**2,ps,stride=1,padding=p)
        h=min(lm.shape[2],lm2.shape[2],x.shape[2]); w=min(lm.shape[3],lm2.shape[3],x.shape[3])
        return (lm2[:,:,:h,:w]-lm[:,:,:h,:w]**2).clamp(min=0)
    def forward(s,p,t):
        l=torch.tensor(0.0,device=p.device)
        if s.wc>0: l=l+s.wc*F.l1_loss(torch.sqrt(s.lv(p,s.ps)+1e-6),torch.sqrt(s.lv(t,s.ps)+1e-6))
        if s.wr>0: B=p.shape[0]; l=l+s.wr*sum((p[i].min()-t[i].min()).abs()+(p[i].max()-t[i].max()).abs() for i in range(B))/B
        return l

class CombinedLoss(nn.Module):
    def __init__(s,*losses): super().__init__(); s.losses=nn.ModuleList(losses)
    def forward(s,p,t): return sum(l(p,t) for l in s.losses)

class OHEMWrapper(nn.Module):
    def __init__(s,c,top_k=0.7): super().__init__(); s.c=c; s.tk=top_k
    def forward(s,p,t,use_ohem=True):
        if not use_ohem or p.size(0)<=1: return s.c(p,t)
        ls=torch.stack([s.c(p[i:i+1],t[i:i+1]) for i in range(p.size(0))])
        return torch.topk(ls,max(1,int(s.tk*len(ls)))).values.mean()

def mixup_data(x,y,alpha=0.2):
    lam=max(np.random.beta(alpha,alpha),1-np.random.beta(alpha,alpha)) if alpha>0 else 1.0
    idx=torch.randperm(x.size(0),device=x.device); return lam*x+(1-lam)*x[idx],lam*y+(1-lam)*y[idx],lam

def split_by_phantom(ds,ne=3,ratios=(0.8,0.1,0.1),seed=42):
    tot=len(ds); nph=tot//ne; assert nph*ne==tot
    rng=np.random.RandomState(seed); pids=np.arange(nph); rng.shuffle(pids)
    nt=int(np.floor(ratios[0]*nph)); nv=int(np.floor(ratios[1]*nph))
    def ti(ps): return [i for p in ps for i in range(p*ne,(p+1)*ne)]
    tr,va,te=pids[:nt],pids[nt:nt+nv],pids[nt+nv:]
    print(f"Synth split: {len(tr)} ph ({len(ti(tr))}) train, {len(va)} ({len(ti(va))}) val, {len(te)} ({len(ti(te))}) test")
    return Subset(ds,ti(tr)),Subset(ds,ti(va)),Subset(ds,ti(te))

def filter_hyperechoic(subset, num_echo_types=3):
    orig=subset.indices; filt=[i for i in orig if i%num_echo_types==0]
    print(f"  Filtered {len(orig)} -> {len(filt)} (hyperechoic only)")
    return Subset(subset.dataset, filt)

def compute_metrics(pn,gn):
    pn=np.clip(pn,0,1); gn=np.clip(gn,0,1); p2=(pn*255).astype(np.uint8); g2=(gn*255).astype(np.uint8)
    mse=np.mean((pn-gn)**2); mae=np.mean(np.abs(pn-gn))
    return {'psnr':peak_signal_noise_ratio(g2,p2,data_range=255),'ssim':structural_similarity(g2,p2,data_range=255,channel_axis=None),
            'vif':vifp(g2,p2),'ncc':compute_ncc(gn,pn),'mse':float(mse),'rmse':float(np.sqrt(mse)),'mae':float(mae),
            'contrast_ratio':float(np.std(pn)/(np.std(gn)+1e-8)),'brightness_diff':float(np.mean(pn)-np.mean(gn)),
            'range_ratio':float((pn.max()-pn.min())/(gn.max()-gn.min()+1e-8))}

METRIC_GROUPS={'Image Quality':['psnr','ssim','vif','ncc'],'Pixel Error':['mse','rmse','mae'],'Contrast':['contrast_ratio','brightness_diff','range_ratio']}
HISTOGRAM_METRICS={'ncc':{'xlabel':'NCC','good_thresh':0.8,'bad_thresh':0.3},'ssim':{'xlabel':'SSIM','good_thresh':0.7,'bad_thresh':0.3},
    'psnr':{'xlabel':'PSNR (dB)','good_thresh':25,'bad_thresh':15},'mae':{'xlabel':'MAE','good_thresh':None,'bad_thresh':None},
    'contrast_ratio':{'xlabel':'Contrast Ratio','good_thresh':None,'bad_thresh':None},'brightness_diff':{'xlabel':'Brightness Diff','good_thresh':None,'bad_thresh':None}}

def print_metrics(ml,label=""):
    print(f"\n=== {label} ({len(ml)} samples) ===")
    for gn,keys in METRIC_GROUPS.items():
        print(f"\n  {gn}:")
        for k in keys: v=[m[k] for m in ml]; print(f"    {k:20s}: mean={np.mean(v):.4f} std={np.std(v):.4f} min={np.min(v):.4f} max={np.max(v):.4f} med={np.median(v):.4f}")

def evaluate(model,loader,crit,device):
    model.eval(); t,n=0.0,0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any(): continue
            t+=crit(model(x),y).item(); n+=1
    return t/max(n,1)

def log_metric_histogram(vals,mn,mi,label,epoch,sd):
    a=np.array(vals,dtype=np.float64); mv,sv,medv=np.mean(a),np.std(a),np.median(a)
    fig,ax=plt.subplots(figsize=(10,5)); _,bins,patches=ax.hist(a,bins=30,alpha=0.7,color='#4C72B0',edgecolor='white')
    for p,le in zip(patches,bins[:-1]):
        if le<mv-sv: p.set_facecolor('#E74C3C')
        elif le>mv+sv: p.set_facecolor('#2ECC71')
    ax.axvline(mv,color='black',linewidth=2,label=f'Mean={mv:.4f}'); ax.axvline(medv,color='orange',linewidth=2,linestyle='--',label=f'Median={medv:.4f}')
    ax.axvspan(mv-sv,mv+sv,alpha=0.15,color='gray',label=f'±1std({sv:.4f})')
    gt,bt=mi.get('good_thresh'),mi.get('bad_thresh')
    if gt: ax.axvline(gt,color='green',linewidth=1.5,linestyle=':')
    if bt: ax.axvline(bt,color='red',linewidth=1.5,linestyle=':')
    ax.set_xlabel(mi.get('xlabel',mn)); ax.set_ylabel('Count')
    ax.set_title(f'{label} — {mn.upper()} (E{epoch})\nMean={mv:.4f} Std={sv:.4f}')
    ax.legend(fontsize=9); m=max(4*sv,0.01); ax.set_xlim(mv-m,mv+m); plt.tight_layout()
    path=os.path.join(sd,f"{mn}_hist_e{epoch}.png"); plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close(); return path

def log_all_histograms(am,label,epoch,sd):
    os.makedirs(sd,exist_ok=True); wl={}
    for mn,mi in HISTOGRAM_METRICS.items():
        v=[m[mn] for m in am]; a=np.array(v); p=log_metric_histogram(v,mn,mi,label,epoch,sd)
        wl[f"{label}/{mn}_histogram"]=wandb.Image(p); wl[f"{label}/{mn}_mean"]=float(np.mean(a)); wl[f"{label}/{mn}_std"]=float(np.std(a)); wl[f"{label}/{mn}_median"]=float(np.median(a))
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for ax,mn in zip(axes.flat,['ncc','ssim','psnr','mae','contrast_ratio','brightness_diff']):
        v=np.array([m[mn] for m in am]); ax.hist(v,bins=25,alpha=0.7,color='#4C72B0',edgecolor='white')
        ax.axvline(np.mean(v),color='black',linewidth=2); ax.axvline(np.median(v),color='orange',linewidth=1.5,linestyle='--')
        ax.set_title(f'{mn.upper()}\nmean={np.mean(v):.4f} med={np.median(v):.4f}',fontsize=10)
    plt.suptitle(f'{label} — Summary (E{epoch})',fontsize=14); plt.tight_layout()
    sp=os.path.join(sd,f"summary_e{epoch}.png"); plt.savefig(sp,dpi=150,bbox_inches='tight'); plt.close()
    wl[f"{label}/summary"]=wandb.Image(sp); wandb.log(wl)

def run_test(model,loader,device,sd,label="Test",log_wandb=True,epoch=None,test_files=None):
    os.makedirs(sd,exist_ok=True); os.makedirs(os.path.join(sd,"images"),exist_ok=True)
    model.eval(); am,ap,ag,wi=[],[],[],[]
    with torch.no_grad():
        for idx,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device); pred=model(x)
            pn=np.clip(pred.squeeze().cpu().numpy(),0,1); gn=np.clip(y.squeeze().cpu().numpy(),0,1)
            m=compute_metrics(pn,gn); am.append(m); ap.append(pn); ag.append(gn)
            fig,(a1,a2)=plt.subplots(1,2,figsize=(10,5))
            a1.imshow(gn,cmap='viridis'); a1.set_title('Ground Truth',fontsize=14); a1.axis('off')
            a2.imshow(pn,cmap='viridis'); a2.set_title('Prediction',fontsize=14); a2.axis('off')
            plt.tight_layout(); ip=os.path.join(sd,"images",f"result_{idx:03d}.png")
            plt.savefig(ip,dpi=200,bbox_inches='tight'); plt.close()
            if log_wandb: wi.append(wandb.Image(ip,caption=f"#{idx} NCC={m['ncc']:.3f} SSIM={m['ssim']:.3f} CR={m['contrast_ratio']:.2f}"))
    np.save(os.path.join(sd,"all_preds.npy"),np.stack(ap)); np.save(os.path.join(sd,"all_gts.npy"),np.stack(ag))
    np.savez(os.path.join(sd,"per_sample_metrics.npz"),**{k:np.array([m[k] for m in am]) for k in am[0]})
    print_metrics(am,label)
    for rm in ['ncc','ssim']:
        arr=np.array([m[rm] for m in am]); worst=np.argsort(arr)[:10]
        print(f"\n  Worst 10 by {rm.upper()}:")
        for r,w in enumerate(worst): fn=test_files[w] if test_files else f"idx {w}"; mx=am[w]; print(f"    {r+1}. idx={w} {rm}={arr[w]:.3f} NCC={mx['ncc']:.3f} SSIM={mx['ssim']:.3f} CR={mx['contrast_ratio']:.2f} file={fn}")
    if log_wandb:
        s={};
        for k in am[0]: v=[m[k] for m in am]; s[f"{label}/{k}_mean"]=float(np.mean(v)); s[f"{label}/{k}_std"]=float(np.std(v)); s[f"{label}/{k}_median"]=float(np.median(v))
        wandb.log(s)
        if wi: wandb.log({f"{label}_predictions":wi})
        log_all_histograms(am,label,epoch or 0,os.path.join(sd,"histograms"))
    return am

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── PHASE 1 ──
    wandb.init(project=WANDB_PROJECT,name=f"fullmodel_p1_{SIM_TYPE}",config={"model":"FullModel (48M)","sim_type":SIM_TYPE,"p1_epochs":PHASE1_EPOCHS,"kwave_hyper_only":SIM_TYPE=="kwave"})
    syn_tr_ds=TransformedSyntheticDataset(synthetic_images,synthetic_labels,augment=True)
    syn_ev_ds=TransformedSyntheticDataset(synthetic_images,synthetic_labels,augment=False)
    syn_tr,_,_=split_by_phantom(syn_tr_ds,NUM_ECHO_TYPES,seed=42)
    _,syn_va,syn_te=split_by_phantom(syn_ev_ds,NUM_ECHO_TYPES,seed=42)
    if SIM_TYPE=="kwave":
        print("\nFiltering to hyperechoic only:")
        syn_tr=filter_hyperechoic(syn_tr,NUM_ECHO_TYPES); syn_va=filter_hyperechoic(syn_va,NUM_ECHO_TYPES); syn_te=filter_hyperechoic(syn_te,NUM_ECHO_TYPES)
    str_l=DataLoader(syn_tr,batch_size=PHASE1_BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
    sva_l=DataLoader(syn_va,batch_size=PHASE1_BATCH_SIZE,shuffle=False)
    ste_l=DataLoader(syn_te,batch_size=1,shuffle=False)

    model=FullModel()
    def iw(m):
        if isinstance(m,(nn.Conv1d,nn.Conv2d)): nn.init.xavier_uniform_(m.weight);
        if hasattr(m,'bias') and m.bias is not None: nn.init.zeros_(m.bias)
    model.apply(iw); model=model.to(device); tp=sum(p.numel() for p in model.parameters())
    print(f"Model: {tp:,} ({tp/1e6:.1f}M)")

    p1c=BalancedLoss(weight_l1=1.0,weight_ssim=0.5,weight_gdl=0.5,weight_tv=0.01,weight_mean=0.1)
    opt=torch.optim.Adam(model.parameters(),lr=PHASE1_START_LR)
    sch=OneCycleLR(opt,max_lr=PHASE1_MAX_LR,total_steps=len(str_l)*PHASE1_EPOCHS,pct_start=0.1,anneal_strategy='cos',cycle_momentum=False)
    print(f"\n{'='*60}\n  P1: {SIM_TYPE} ({len(syn_tr)} samples)\n{'='*60}")
    bv,bep1,esc=float('inf'),0,0; os.makedirs(P1_SAVE_DIR,exist_ok=True)

    for ep in range(PHASE1_EPOCHS):
        model.train(); tl=0.0
        for x,y in tqdm(str_l,desc=f"P1 E{ep+1}"):
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=p1c(model(x),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5); opt.step(); sch.step(); tl+=loss.item()
        tl/=len(str_l); vl=evaluate(model,sva_l,p1c,device)
        wandb.log({"phase":1,"epoch":ep+1,"p1/train_loss":tl,"p1/val_loss":vl,"p1/lr":opt.param_groups[0]['lr']})
        if ep%5==0:
            model.eval()
            with torch.no_grad():
                for x,y in sva_l: x,y=x.to(device),y.to(device); pred=model(x); break
                wandb.log({"p1/prediction":wandb.Image(np.clip(pred[0].squeeze().cpu().numpy(),0,1)),"p1/target":wandb.Image(np.clip(y[0].squeeze().cpu().numpy(),0,1))})
        if vl<bv: bv=vl; bep1=ep; esc=0; torch.save(model.state_dict(),os.path.join(P1_SAVE_DIR,f"p1_epoch{ep}.pth"))
        else:
            esc+=1
            if esc>=PHASE1_PATIENCE: print(f"Early stop at {ep+1}"); break
        print(f"P1 E{ep+1}: Train={tl:.4f} Val={vl:.4f}")

    p1ckpt=os.path.join(P1_SAVE_DIR,f"p1_epoch{bep1}.pth"); print(f"\nBest P1: epoch {bep1}")
    model.load_state_dict(torch.load(p1ckpt,map_location=device,weights_only=True))
    print("\nP1 synth test..."); run_test(model,ste_l,device,f"results/fullmodel_{SIM_TYPE}_p1_synth","P1 Synthetic",epoch=bep1)
    wandb.finish(); print("P1 done.\n")

    # ── PHASE 2 ──
    cf=ClinicalLabelledDataset(clinical_rf_folder,clinical_label_folder,mask_folder=clinical_mask_folder,augment=False)
    nc=len(cf); rng=np.random.RandomState(99); idx=np.arange(nc); rng.shuffle(idx)
    atr=idx[:nc-CLINICAL_VAL_COUNT-CLINICAL_TEST_COUNT].tolist()
    vi=idx[nc-CLINICAL_VAL_COUNT-CLINICAL_TEST_COUNT:nc-CLINICAL_TEST_COUNT].tolist()
    ti=idx[nc-CLINICAL_TEST_COUNT:].tolist(); tf=[cf.rf_files[i] for i in ti]
    ca=ClinicalLabelledDataset(clinical_rf_folder,clinical_label_folder,mask_folder=clinical_mask_folder,augment=True)
    vl_l=DataLoader(Subset(cf,vi),batch_size=1,shuffle=False,num_workers=0)
    tl_l=DataLoader(Subset(cf,ti),batch_size=1,shuffle=False,num_workers=0)

    for cap in TRAIN_CAPS:
        rn=f"fullmodel_{SIM_TYPE}_p2_{cap}pt"
        print(f"\n{'#'*60}\n  P2: {SIM_TYPE} -> {cap} clinical\n{'#'*60}")
        wandb.init(project=WANDB_PROJECT,name=rn,reinit=True,config={"model":"FullModel (48M)","sim_type":SIM_TYPE,"p1_checkpoint":p1ckpt,"train_cap":cap,"p2_epochs":PHASE2_EPOCHS,"kwave_hyper_only":SIM_TYPE=="kwave"})
        tr_l=DataLoader(Subset(ca,atr[:cap]),batch_size=PHASE2_BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
        print(f"Clinical: {cap} train, {len(vi)} val, {len(ti)} test")

        model=FullModel().to(device); model.load_state_dict(torch.load(p1ckpt,map_location=device,weights_only=True))
        print(f"Loaded P1: {p1ckpt}")
        print("\n--- Before FT ---")
        run_test(model,tl_l,device,f"results/fullmodel_{SIM_TYPE}_{cap}pt_before_ft","P1 on Clinical (no FT)",epoch=0,test_files=tf)
        if os.path.isdir("data/ClinicalData"):
            from utils.predict_clinical_pairs import predict_clinical_pairs
            predict_clinical_pairs(model,device,clinical_folder="data/ClinicalData",mask_folder=clinical_mask_folder,save_dir=f"results/fullmodel_{SIM_TYPE}_{cap}pt_pairs_before",wandb_label="ClinicalData (before FT)",log_wandb=True)

        bl=BalancedLoss(weight_l1=WEIGHT_L1,weight_ssim=WEIGHT_SSIM,weight_gdl=WEIGHT_GDL,weight_tv=WEIGHT_TV,weight_mean=WEIGHT_MEAN)
        cl2=ContrastAwareLoss(wc=WEIGHT_CONTRAST,wr=WEIGHT_RANGE); pl=PerceptualLoss(weight=WEIGHT_PERCEPTUAL).to(device)
        comb=CombinedLoss(bl,cl2,pl); oh=OHEMWrapper(comb,top_k=OHEM_TOP_K); vcr=comb
        opt=torch.optim.Adam(model.parameters(),lr=PHASE2_LR); sch=CosineAnnealingWarmRestarts(opt,T_0=T_0,T_mult=T_MULT,eta_min=1e-6)
        bv2,bep2=float('inf'),0; smd=f"model/best_fullmodel_{SIM_TYPE}_{cap}pt"; os.makedirs(smd,exist_ok=True)

        for ep in range(1,PHASE2_EPOCHS+1):
            model.train(); tls,nb=0.0,0; uo=USE_OHEM and(ep>OHEM_WARMUP)
            for x,y in tqdm(tr_l,desc=f"P2 E{ep} ({cap}pt)"):
                x,y=x.to(device),y.to(device)
                if torch.isnan(x).any() or torch.isinf(x).any(): continue
                if USE_MIXUP and ep>2: x,y,_=mixup_data(x,y,alpha=MIXUP_ALPHA)
                opt.zero_grad(); loss=oh(model(x),y,use_ohem=uo)
                if torch.isnan(loss): continue
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5); opt.step(); tls+=loss.item(); nb+=1
            sch.step(ep); tl2=tls/max(nb,1); vl2=evaluate(model,vl_l,vcr,device)
            wandb.log({"epoch":ep,"p2/train_loss":tl2,"p2/val_loss":vl2,"p2/lr":opt.param_groups[0]['lr']})
            if ep%10==0 or ep==1:
                model.eval(); em=[]; sl=False
                with torch.no_grad():
                    for xt,yt in vl_l:
                        xt,yt=xt.to(device),yt.to(device); pt=model(xt)
                        if not sl: wandb.log({"p2/prediction":wandb.Image(np.clip(pt[0].squeeze().cpu().numpy(),0,1)),"p2/target":wandb.Image(np.clip(yt[0].squeeze().cpu().numpy(),0,1))}); sl=True
                        em.append(compute_metrics(np.clip(pt.squeeze().cpu().numpy(),0,1),np.clip(yt.squeeze().cpu().numpy(),0,1)))
                hd=f"results/fullmodel_{SIM_TYPE}_{cap}pt_hist"; os.makedirs(hd,exist_ok=True); log_all_histograms(em,"p2/during_training",ep,hd)
            if vl2<bv2: bv2=vl2; bep2=ep; torch.save(model.state_dict(),os.path.join(smd,f"p2_epoch{ep}.pth"))
            print(f"P2 E{ep}: Train={tl2:.4f} Val={vl2:.4f} OHEM={'ON' if uo else 'off'} LR={opt.param_groups[0]['lr']:.2e}")

        bp=os.path.join(smd,f"p2_epoch{bep2}.pth"); print(f"\nBest P2: epoch {bep2}")
        model.load_state_dict(torch.load(bp,map_location=device,weights_only=True))
        print(f"\n{'='*60}\n  Final: {SIM_TYPE} {cap}pt\n{'='*60}")
        run_test(model,tl_l,device,f"results/fullmodel_{SIM_TYPE}_{cap}pt_final",f"Clinical ({SIM_TYPE} {cap}pt)",epoch=bep2,test_files=tf)
        if os.path.isdir("data/ClinicalData"):
            predict_clinical_pairs(model,device,clinical_folder="data/ClinicalData",mask_folder=clinical_mask_folder,save_dir=f"results/fullmodel_{SIM_TYPE}_{cap}pt_pairs_after",wandb_label="ClinicalData (after FT)",log_wandb=True)
        wandb.finish(); print(f"Done: {rn}\n")
    print("\nAll complete!")

if __name__=="__main__": main()