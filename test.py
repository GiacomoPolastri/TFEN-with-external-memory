import fiftyone as fo
from Data.AMdataset import GetDataset
from fiftyone import ViewField as F

print (fo.list_datasets())

trainset = fo.load_dataset('Aerial_Maritime_trainset')
fo.pprint(trainset.stats(include_media=True))

trainset.delete()