import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def printOutputDist(df_path,outputfolder):
    classes = ["In_Arch","In_Constr","Out_Constr","Out_Urban","Forest"]
    figname = df_path.split("/")[-1].split(".")[0]
    figname = figname.split("_")[1]
    df = pd.read_csv(df_path)
    mean_list = []
    var_list = []
    for hailoClass in classes:
        mean_list.append(df.loc[:, hailoClass].mean())
        var_list.append(df.loc[:, hailoClass].std())
    
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    x = np.linspace(0,1, 1000)
    for i,(mean,std) in enumerate(zip(mean_list,var_list)):
        plt.plot(x, stats.norm.pdf(x, mean, std),label = f"{classes[i]} Mean:{mean:.3} Std:{std:.3}")
        # plt.fill(x,stats.norm.pdf(x, mean, std))
        
    for mean in mean_list:
        plt.axvline(mean,linestyle='dashed') 
        
    plt.title(figname)
    plt.xlabel('X')
    plt.ylabel('Prob')
    plt.legend()
    plt.grid(True)
    plt.savefig(outputfolder + "/" +  figname, dpi=600, bbox_inches='tight')
    plt.clf()
    
if __name__ == "__main__":
    
    df_path = "Evaluation/Data/Hailo/pred_RN101_5patches_5scentens.csv"
    outPath = "Evaluation/Data/temp"
    
    printOutputDist(df_path,outPath)