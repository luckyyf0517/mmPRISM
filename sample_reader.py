import pickle
from glob import glob
from tqdm import tqdm
from make_video import make_video

caption_path = 'data/csl-daily/sentence_label/csl2020ct_v2.pkl'
with open(caption_path, 'rb') as f:
    data = pickle.load(f)
    
# Create mapping from video ID to annotation
caption_dict = {}
for item in tqdm(data['info']):
    # Construct video ID: name_signer_time
    video_id = item['name']
    # Use character-level annotation as caption
    caption = ''.join(item['label_char'])
    caption_dict[video_id] = caption

from IPython import embed; embed()
