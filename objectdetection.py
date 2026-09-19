import json
import glob
import os
import numpy as np
import pandas as pd
from imageai.Detection import ObjectDetection
from tqdm import tqdm



def load_paths():
    datapath=''
    output_path=''
    jsonpath=os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset_path.txt')

    with open(jsonpath, 'r') as file:
        json_dict=json.loads(file.read())
        datapath=json_dict['data_path']
        output_path=json_dict['output_path']
    return datapath, output_path

def load_images_paths(dataset_paths):
    images_paths=[]
    for path in dataset_paths:
        images_paths+=glob.glob(os.path.join(path,'*.*'))
    return images_paths

def load_images(json_path):
    
    with open(json_path,'r') as file:
        path_json=json.loads(file.read())

    image_data=pd.read_csv(path_json['data'])
    
    images_path=load_images_paths(list(path_json.values()))
    images_names=[os.path.basename(im) for im in images_path]

    columns=['path','filename']
    data={'path':images_path,'filename':images_names}
    df=pd.DataFrame(columns=columns,data=data)

    df2=pd.merge(df,image_data,on='filename')
    
    final_image_path=np.array(df2.path)
    return final_image_path

# def load_images(datapath,sample_size=10):
#     image_path=np.array(glob.glob(os.path.join(datapath,'*.*')))
#     sampled_idx=np.random.choice(np.arange(len(image_path)),sample_size,replace=False)
#     return image_path[sampled_idx],sampled_idx

def create_output_paths(output_path,image_paths):
    output_names=[os.path.basename(im).split('.')[0] for im in image_paths]
    output_paths=[]
    for name in output_names:
        name = name+'_output.jpg'
        path=os.path.join(output_path,name)
        output_paths.append(path)
    return output_paths

# def create_output_paths(output_path,idx,datapath):
# output_names=np.array(os.listdir(datapath))[idx]
# output_paths=[]
# for name in output_names:
#     name = name.split('.')[0]
#     name = name+'_output.jpg'
#     path=os.path.join(output_path,name)
#     output_paths.append(path)
# return output_paths

def load_catigories(path=os.path.join(os.getcwd(),'catgores.txt')):
    with open(path,'r') as file:
        catgores_dict=json.loads(file.read())
    return catgores_dict


def main():
   
    datapath,output_path=load_paths()
    json_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'all_dataset_path.json')
    input_paths=load_images(json_path=json_path)
    output_paths=create_output_paths(output_path=output_path,image_paths=input_paths)


    execution_path = os.path.dirname(os.path.abspath(__file__))
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "model.h5"))

    detector.loadModel()

    catigories=load_catigories()

    data=[]
    for input,output in tqdm(zip(input_paths, output_paths)):
        detections = detector.detectObjectsFromImage(input_image=input,output_image_path=output)
        object_names=[]
        object_probs=[]
        
        object_supercatigories=list(np.unique(list(catigories.values())))

        for eachObject in detections:
            # print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
            object_names.append(eachObject['name'])
            object_probs.append(eachObject['percentage_probability'])
            supercatigory=catigories[eachObject['name']]
            object_supercatigories.append(supercatigory)

        object_supercatigories=np.array(object_supercatigories)

        _, object_count=np.unique(object_supercatigories,return_counts=True)
        object_count-=1
        num_object=len(object_names)
        data.append([os.path.basename(input),num_object,object_names,object_supercatigories[-num_object:],object_probs]+list(object_count))

    columns = ['image_name','num_objects','objects','super_catigory','probabilities']+list(np.unique(list(catigories.values())))
    df=pd.DataFrame(columns=columns,data=data)
    csv_path=os.path.join(os.getcwd(),'images_csv.csv')

    df.to_csv(csv_path)



if __name__ == '__main__':
    main()
    