import json
import pandas as pd
import os
import glob
import numpy as np
 
def load_images(dataset_paths):
    images_paths=[]
    for path in dataset_paths:
        images_paths+=glob.glob(os.path.join(path,'*.*'))
    return images_paths



def main():
    json_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'all_dataset_path.json')
    with open(json_path,'r') as file:
        path_json=json.loads(file.read())

    image_data=pd.read_csv(path_json['data'])
    
    images_path=load_images(list(path_json.values()))
    images_names=[os.path.basename(im) for im in images_path]

    columns=['path','filename']
    data={'path':images_path,'filename':images_names}
    df=pd.DataFrame(columns=columns,data=data)

    df2=pd.merge(df,image_data,on='filename')
    
    final_image_path=np.array(df2.path)
    print(final_image_path)

if __name__=='__main__':
    main()