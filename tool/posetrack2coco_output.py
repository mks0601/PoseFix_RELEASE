import json
import glob
import numpy as np

filenames = glob.glob('*.json')
combined_annot_path = 'result.json'
combined_annot = []
posetrack18_ignore_kps = [3,4] # l_ear, r_ear index

for i in range(len(filenames)):

    with open(filenames[i]) as f:
        annot = json.load(f)['annotations']

    for j in range(len(annot)):
        scores = annot[j]['scores']
        scores = np.delete(scores,posetrack18_ignore_kps) # l_ear and r_ear are not annotated
        annot[j]['score'] = np.mean(scores)

    combined_annot += annot
    
with open(combined_annot_path, 'w') as f:
    json.dump(combined_annot, f)

