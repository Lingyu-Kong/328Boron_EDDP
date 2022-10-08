import argparse
import wandb
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from graph.traj2graph import Traj2Graph
from model.eddp import EDDP
from normalize import Normalizer_Per_Atom

parser=argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="B3_shake")
parser.add_argument("--test_data", type=str, default="B8_shake")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--pmin", type=float, default=2.0)
parser.add_argument("--pmax", type=float, default=10.0)
parser.add_argument("--m", type=int, default=6)
parser.add_argument("--cutoff", type=float, default=3.75)
parser.add_argument("--supercell", type=int, default=3)
parser.add_argument("--use_normalize", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--wandb",action="store_true")
args=parser.parse_args()

if args.wandb:
    wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
    wandb.init(project="328Boron_EDDP",name=args.train_data+"->"+args.test_data,config=args)
    
print(args)

M=args.m
FEATURE_DIM=1+M+M**2
SUPERCELL=np.array([[-(args.supercell//2), args.supercell//2+1]]*3, dtype=np.int32)

trajloader=Traj2Graph(args.cutoff, args.pmin, args.pmax, M, SUPERCELL)
if args.train_data!=args.test_data:
    training_data,train_loading_time=trajloader.load(args.train_data, shuffle=True,)
    testing_data,test_loading_time=trajloader.load(args.test_data, shuffle=True,length=int(len(training_data)*0.1))
else:
    data,loading_time=trajloader.load(args.train_data, shuffle=True)
    training_data=data[:int(len(data)*0.9)]
    train_loading_time=loading_time*0.9
    testing_data=data[int(len(data)*0.9):]
    test_loading_time=loading_time*0.1

TRAIN_NUM_ATOMS=training_data[0].num_atoms
TEST_NUM_ATOMS=testing_data[0].num_atoms
    
print("Training data size: {},  Num_atoms: {},  Time: {}".format(len(training_data),TRAIN_NUM_ATOMS,train_loading_time))
print("Testing data size: {},  Num_atoms: {},  Time: {}".format(len(testing_data),TEST_NUM_ATOMS,test_loading_time))

if args.use_normalize:
    normalizer=Normalizer_Per_Atom()
    energy_mean,energy_std,training_data=normalizer.normalize(training_data)
    _,_,testing_data=normalizer.normalize(testing_data,mean=energy_mean,std=energy_std)
else:
    energy_mean=0.0
    energy_std=1.0
    
print("Energy mean: {},  Energy std: {}".format(energy_mean,energy_std))

trainloader=DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
testloader=DataLoader(testing_data, batch_size=args.batch_size, shuffle=False)

device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

model_args={
    "input_size":FEATURE_DIM,
    "output_sizes":[64,32,1],
    "use_layer_norm":False,
    "activation":nn.SiLU,
    "dropout":0.0,
    "layernorm_before":False,
    "use_bn":True,
    "device":device
}

if __name__=="__main__":
    
    model=EDDP(**model_args)
    
    for epoch in range(args.num_epochs):
        train_loss=[]
        test_loss=[]
        train_mae=[]
        test_mae=[]
        start_time=time.time()
        for i,graph_batch in enumerate(trainloader):
            energies=(graph_batch.energy).to(device)
            features=(graph_batch.feature).to(device)
            node_batch=graph_batch.batch.to(device)
            num_graphs=graph_batch.num_graphs
            energies_pred=model(features, node_batch, num_graphs)
            loss=torch.mean((energies-energies_pred)**2)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            train_loss.append(loss.item())
            mae=torch.mean(torch.abs(energies-energies_pred))
            train_mae.append(mae.item())
        for i,graph_batch in enumerate(testloader):
            energies=(graph_batch.energy).to(device)
            features=graph_batch.feature.to(device)
            node_batch=graph_batch.batch.to(device)
            num_graphs=graph_batch.num_graphs
            energies_pred=model(features, node_batch, num_graphs)
            loss=torch.mean((energies-energies_pred)**2)
            test_loss.append(loss.item())
            mae=torch.mean(torch.abs(energies-energies_pred))
            test_mae.append(mae.item())
        end_time=time.time()
        train_loss=np.mean(train_loss)*energy_std**2
        test_loss=np.mean(test_loss)*energy_std**2
        train_mae=np.mean(train_mae)*energy_std
        test_mae=np.mean(test_mae)*energy_std
        if args.wandb:
            wandb.log({"train_loss":train_loss,"test_loss":test_loss,"train_mae":train_mae,"test_mae":test_mae})
        print("Epoch: {},  Train loss: {},  Test loss: {}".format(epoch,train_loss,test_loss))
        print("Train MAE: {},  Test MAE: {},  Time: {}".format(train_mae,test_mae,end_time-start_time))
        print("=====================================================================================================")
    
    ## Performance
    energies=[]
    energies_pred=[]
    for i,graph_batch in enumerate(trainloader):
        energies=energies+(graph_batch.energy*energy_std+energy_mean*TRAIN_NUM_ATOMS).tolist()
        features=graph_batch.feature.to(device)
        node_batch=graph_batch.batch.to(device)
        num_graphs=graph_batch.num_graphs
        energies_pred=energies_pred+model.eval(features, node_batch, num_graphs,mean=energy_mean*TRAIN_NUM_ATOMS,std=energy_std).tolist()
    for energy in energies:
        energy=energy*energy_std+energy_mean*TRAIN_NUM_ATOMS
    min_energy=min([min(energies),min(energies_pred)])
    plt.figure()
    plt.scatter(energies,energies_pred,marker=".")
    # plt.xlim((min_energy, min_energy+10))
    # plt.ylim((min_energy, min_energy+10))
    if args.wandb:
        wandb.log({"train_scatter":wandb.Image(plt)})
    
    energies=[]
    energies_pred=[]
    for i,graph_batch in enumerate(testloader):
        energies=energies+(graph_batch.energy*energy_std+energy_mean*TEST_NUM_ATOMS).tolist()
        features=graph_batch.feature.to(device)
        node_batch=graph_batch.batch.to(device)
        num_graphs=graph_batch.num_graphs
        energies_pred=energies_pred+model.eval(features, node_batch, num_graphs,mean=energy_mean*TEST_NUM_ATOMS,std=energy_std).tolist()
    min_energy=min([min(energies),min(energies_pred)])
    plt.figure()
    plt.scatter(energies,energies_pred,marker=".")
    # plt.xlim((min_energy, min_energy+10))
    # plt.ylim((min_energy, min_energy+10))
    if args.wandb:
        wandb.log({"test_scatter":wandb.Image(plt)})
    
    if args.save_model:
        model.save_model("model.pt")
        if args.wandb:
            wandb.save("model.pt")
            
            
            

