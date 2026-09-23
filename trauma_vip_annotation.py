# trauma_vip_annotation.py
# End-to-end YOLO tracking, VIP detection, annotations JSON and COCO export
import os, cv2, json, numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

INPUT_ROOT=r"D:\Project - 05 Dec 2025\Trauma_Dataset_Input"
OUTPUT_ROOT=r"D:\Project - 05 Dec 2025\Trauma_Dataset_Output"
MODEL_PATH='yolo11n.pt'
CONF_THRESHOLD=0.25
SAVE_FRAMES=True
AREA_WEIGHT=0.6
CENTER_WEIGHT=0.2
AGE_WEIGHT=0.2
SCORE_SMOOTHING=0.9
VIP_HYSTERESIS=1.3

model=YOLO(MODEL_PATH)

def create_video_label(video_path):
    p=video_path.lower()
    if 'no trauma clips' in p: return 'no_trauma'
    if 'trauma clips' in p: return 'trauma'
    return 'unknown'


def process_video(video_path, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    frames_dir=os.path.join(output_dir,'frames')
    os.makedirs(frames_dir,exist_ok=True)
    video_name=Path(video_path).stem

    cap=cv2.VideoCapture(video_path)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    out_video=os.path.join(output_dir,Path(video_path).name)
    writer=cv2.VideoWriter(out_video,cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))

    annotation_json={
        'video_id':video_name,
        'video_label':create_video_label(video_path),
        'video_comment':'',
        'context':'',
        'frames':{},
        'frame_dims':{}
    }

    coco={
      'info':{'description':f'Annotated objects for {video_name}','video_id':video_name,'video_label':create_video_label(video_path),'video_comment':'','context':''},
      'images':[],
      'annotations':[],
      'categories':[{'id':1,'name':'person_of_interest'},{'id':2,'name':'person'},{'id':3,'name':'other'}]
    }

    vip_scores=defaultdict(float)
    track_age=defaultdict(int)
    vip_locked_id=None
    frame_idx=1
    ann_id=1
    frame_center_x=width/2
    frame_center_y=height/2
    max_area=width*height
    diagonal=np.sqrt(width**2+height**2)

    for res in model.track(source=video_path,stream=True,persist=True,classes=[0]):
        frame=res.orig_img.copy()
        current_scores={}
        dets=[]

        if res.boxes.id is not None:
            boxes=res.boxes.xyxy.cpu().numpy()
            ids=res.boxes.id.cpu().numpy().astype(int)
            confs=res.boxes.conf.cpu().numpy()
            for box,tid,conf in zip(boxes,ids,confs):
                if conf<CONF_THRESHOLD: continue
                x1,y1,x2,y2=map(int,box)
                cx=(x1+x2)/2; cy=(y1+y2)/2
                track_age[tid]+=1
                area=((x2-x1)*(y2-y1))/max_area
                center=1-(np.sqrt((cx-frame_center_x)**2+(cy-frame_center_y)**2)/diagonal)
                age=min(track_age[tid]/300.0,1.0)
                imp=AREA_WEIGHT*area+CENTER_WEIGHT*center+AGE_WEIGHT*age
                vip_scores[tid]=SCORE_SMOOTHING*vip_scores[tid]+(1-SCORE_SMOOTHING)*imp
                current_scores[tid]=vip_scores[tid]
                dets.append((x1,y1,x2,y2,tid,float(conf)))

        if current_scores:
            cand=max(current_scores,key=current_scores.get)
            if vip_locked_id is None:
                vip_locked_id=cand
            elif current_scores[cand] > current_scores.get(vip_locked_id,0)*VIP_HYSTERESIS:
                vip_locked_id=cand

        frame_name=f'frame_{frame_idx:06d}.jpg'
        objs=[]
        coco['images'].append({'id':frame_idx,'file_name':frame_name,'width':width,'height':height,'behaviours':['normal'],'behaviour':'normal','comment':''})

        for x1,y1,x2,y2,tid,conf in dets:
            is_poi=(tid==vip_locked_id)
            color=(0,0,255) if is_poi else (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
            cv2.putText(frame,('VIP-' if is_poi else 'ID:')+str(tid),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            obj={
             'id':f'track_{tid}','bbox':[float(x1),float(y1),float(x2),float(y2)],'label':'person_of_interest' if is_poi else 'person','behaviours':['normal'],'confirmed':bool(True),'source':'yolo','conf':round(conf,4),'is_poi':bool(is_poi)}
            objs.append(obj)
            w=x2-x1; h=y2-y1
            coco['annotations'].append({'id':ann_id,'image_id':frame_idx,'category_id':1 if is_poi else 2,'bbox':[float(x1),float(y1),float(w),float(h)],'area':float(w*h),'iscrowd':0,'behaviours':['normal'],'behaviour':'normal','object_id':f'track_{tid}','is_poi':bool(is_poi)})
            ann_id+=1

        if objs:
            if SAVE_FRAMES:
                cv2.imwrite(os.path.join(frames_dir,frame_name),frame)
            annotation_json['frame_dims'][frame_name]=[width,height]
            annotation_json['frames'][frame_name]={'behaviours':['normal'],'comment':'','bbox':objs[0]['bbox'],'objects':objs}

        writer.write(frame)
        frame_idx+=1

    writer.release()
    with open(os.path.join(output_dir,f'{video_name}_annotations.json'),'w',encoding='utf-8') as f: json.dump(annotation_json,f,indent=2)
    with open(os.path.join(output_dir,f'{video_name}_coco.json'),'w',encoding='utf-8') as f: json.dump(coco,f,indent=2)


def process_dataset(root_dir):
    exts=('.mp4','.avi','.mov','.mkv')
    for root,dirs,files in os.walk(root_dir):
        if 'output' in root.lower():
            continue
        for file in files:
            if not file.lower().endswith(exts):
                continue
            rel=os.path.relpath(root,root_dir)
            video_name=os.path.splitext(file)[0]
            out_dir=os.path.join(OUTPUT_ROOT,rel,video_name)
            process_video(os.path.join(root,file),out_dir)

if __name__=='__main__':
    process_dataset(INPUT_ROOT)
    print('Finished.')
