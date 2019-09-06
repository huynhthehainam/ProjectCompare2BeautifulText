from Model import SiameseModel


model  = SiameseModel()
model.LoadRawData()
model.ProcessRawData()
model.Train()