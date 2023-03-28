import fiftyone as fo 

dataset_type = fo.types.COCODetectionDataset
dataset_dir = "./Inputs/"

class LoaDataset():
    
    def __init__(self):
        
        self.dataset = self.load_all()
        
    def load_all(self):
        
        trainset = self.loadTrainset()
        validset = self.loadValidset()
        testset = self.loadTestset()
        
        return trainset, validset, testset
    
    def loadTrainset(self):
    
        name = "trainset"
        trainSet_dir = dataset_dir + "train"
        label_path = trainSet_dir + "/_annotations.coco.json"
        
        trainset = fo.Dataset.from_dir(
            data_path = trainSet_dir,
            dataset_type = dataset_type,
            labels_path = label_path,
            name = name,
        )   
        
        trainset.tags.append('train')
        for sample in trainset.iter_samples(progress=True):
            sample.tags.append('train')
            sample.save()
        trainset.save()
        
        trainset.persistent = True
        
        return trainset
    
    def loadValidset(self):
        
        name = "validset"
        validSet_dir = dataset_dir + "valid"
        label_path = validSet_dir + "/_annotations.coco.json"

        validset = fo.Dataset.from_dir(
            data_path = validSet_dir,
            dataset_type = dataset_type,
            labels_path = label_path,
            name = name
        )   
        
        validset.tags.append('valid')
        for sample in validset.iter_samples(progress=True):
            sample.tags.append('valid')
            sample.save()
        validset.save()
        
        validset.persistent = True

        return validset
    
    def loadTestset(self):
        
        name = "testset"
        testSet_dir = dataset_dir + "test"
        label_path = testSet_dir + "/_annotations.coco.json"

        testset = fo.Dataset.from_dir(
            data_path = testSet_dir,
            dataset_type = dataset_type,
            labels_path = label_path,
            name = name
        )   

        testset.tags.append('test')
        for sample in testset.iter_samples(progress=True):
            sample.tags.append('test')
            sample.save()
        testset.save()

        testset.persistent = True
        
        return testset

class GetDataset():
    
    def __init__(self):
        
        self.datasets = self.check_dataset()
        self.train_set = fo.load_dataset("trainset")
        self.valid_set = fo.load_dataset("validset")
        self.test_set = fo.load_dataset("testset")
        
    def check_dataset(self):
        
        dataset = None
        if not fo.dataset_exists("trainset"):
            dataset = LoaDataset()
            
        return dataset

if __name__ == "__main__":
    
    #dataset = LoaDataset().dataset
    #trainset = fo.load_dataset("trainset")
    #print (trainset)
    #dataset = fo.load_dataset("validset")
    #dataset.delete()
    print(fo.list_datasets())