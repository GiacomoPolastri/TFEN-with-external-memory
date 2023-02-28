import fiftyone as fo 

dataset_type = fo.types.COCODetectionDataset
dataset_dir = "./Inputs/Aerial_Maritime.v9-tiled.coco/"

class LoaDataset():
    
    def __init__(self):
        
        self.trainset = self.loadTrainset()
        self.validset = self.loadValidset()
        self.testset = self.loadTestset()
    
    def loadTrainset(self):
    
        name = "Aerial_Maritime_trainset"
        trainSet_dir = dataset_dir + "train"
        label_path = trainSet_dir + "/_annotations.coco.json"
        
        traintest = fo.Dataset.from_dir(
            data_path = trainSet_dir + "/data",
            dataset_type = dataset_type,
            labels_path = label_path,
            name = name,
        )   
        
        traintest.persistent = True
        
        return traintest
    
    def loadValidset(self):
        
        name = "Aerial_Maritime_validset"
        validSet_dir = dataset_dir + "valid"
        label_path = validSet_dir + "/_annotations.coco.json"

        validtest = fo.Dataset.from_dir(
            data_path = validSet_dir,
            dataset_type = dataset_type,
            labels_path = label_path,
            name = name
        )   
        
        validtest.persistent = True

        return validtest
    
    def loadTestset(self):
        
        name = "Aerial_Maritime_testset"
        testSet_dir = dataset_dir + "test"
        label_path = testSet_dir + "/_annotations.coco.json"

        testset = fo.Dataset.from_dir(
            data_path = testSet_dir,
            dataset_type = dataset_type,
            labels_path = label_path,
            name = name
        )   

        testset.persistent = True
        
        return testset

class GetDataset():
    
    def __init__(self):
        
        self.datasets = self.check_dataset()
        self.train_set = fo.load_dataset("Aerial_Maritime_trainset")
        self.valid_set = fo.load_dataset("Aerial_Maritime_validset")
        self.test_set = fo.load_dataset("Aerial_Maritime_testset")
        
    def check_dataset(self):
        
        dataset = None
        if not fo.dataset_exists("Aerial_Maritime_trainset"):
            dataset = LoaDataset()
            
        return dataset

if __name__ == "__main__":
    
    classifications_dataset = fo.load_dataset("Aerial_Maritime.v9")
    
    a = LoaDataset()
    
    session = fo.launch_app(classifications_dataset, desktop=True)
    session.wait()
    
    #if fo.dataset_exists('Aerial_Maritime_trainset'):
    #    print ('yes')
    #    
    #validset = fo.load_dataset('Aerial_Maritime_validset')
    #print (validset)
    
    #session = fo.launch_app(validset, desktop=True)
    #session.wait()