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
            data_path = trainSet_dir,
            dataset_type = dataset_type,
            labels_path = label_path,
            name = name
        )   
        
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
    
    trainset = GetDataset().train_set
    trainset.compute_metadata()
    session = fo.launch_app(trainset, desktop=True)
    session.wait()

    
    #print (LoaDataset().trainset)
    #if fo.dataset_exists('Aerial_Maritime_trainset'):
    #    print ('yes')
    #    
    #validset = fo.load_dataset('Aerial_Maritime_validset')
    #print (validset)
    
    #session = fo.launch_app(validset, desktop=True)
    #session.wait()