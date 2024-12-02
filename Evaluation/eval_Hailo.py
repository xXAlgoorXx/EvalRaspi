import os
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path
from cpuEval_utils import get_max_class_with_threshold,get_trueClass,find_majority_element,printAndSaveHeatmap
from hailoEval_utils import get_pred_hailo,get_throughput_hailo, HailoCLIPImage, HailoCLIPText, getCorrespondingGemm, getCorrespondingTextEmb


# Pathes
outputPath = Path("Evaluation/Data/Hailo")
Dataset5Patch224px = Path("candolle_5patches_224px")
Dataset5Patch = Path("candolle_5patch")
tinyClipModels = Path("tinyClipModels")

hefFolder_path = "Evaluation/resources/hef" # get from for loop
postprocess_path = "Evaluation/resources/json/gemm_layer_RN50x4.json"
textEmbeddings_path = "Evaluation/resources/json/textEmbeddings_RN50x4.json"


def main():
    heffiles = next(os.walk(hefFolder_path), (None, None, []))[2]
    for hef in heffiles:
        hef_path = hefFolder_path + "/" + hef
        postprocess_path = getCorrespondingGemm(hef_path)
        textEmbeddings_path = getCorrespondingTextEmb(hef_path)
        
        hailoImagemodel = HailoCLIPImage(hef_path,postprocess_path)
        hailoTextmodel = HailoCLIPText(textEmbeddings_path)
        modelname = hailoTextmodel.getModelName()
        use5Scentens = hailoTextmodel.getuse5Scentens()
        accuracy_models = []
        
        # Path to csv
        if use5Scentens:
            csv_path_predictions = outputPath / f'pred_{modelname}_5patches_5scentens.csv'
        else:
            csv_path_predictions = outputPath / f'pred_{modelname}_5patches.csv'

        # check if csv already exists
        if os.path.exists(csv_path_predictions):
            df_5patch = get_trueClass(pd.read_csv(csv_path_predictions))
        else:                
            df_pred = get_pred_hailo(Dataset5Patch,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
            df_pred.to_csv(csv_path_predictions, index=False)
            df_5patch = get_trueClass(pd.read_csv(csv_path_predictions))
        df = df_5patch.copy()
        df['y_predIO'] = df.apply(get_max_class_with_threshold, axis=1, threshold=0.8)

        # set the outdoor classes to 0 when the image was classified as indoor 
        # set the indoor classes to 0 when the image was classified as outdoor 
        df.loc[df['y_predIO'] == 'In', ['Out_Constr', 'Out_Urban', 'Forest']] = 0 
        df.loc[df['y_predIO'] == 'Out', ['In_Arch', 'In_Constr']] = 0

        # create the new column y_predIO
        columns = ['In_Arch', 'In_Constr', 'Out_Constr', 'Out_Urban', 'Forest']
        df['y_pred'] = df[columns].idxmax(axis=1)

        # evaluate performance of model
        y_test = df['ClassTrue']
        y_pred = df['y_pred']

        # majority counts
        y_test_s = []
        majority_pred = []

        # iterate through the input array in chunks of 5
        for i in range(0, len(y_test), 5):

            patches = y_test[i:i+5]
            majority_element = find_majority_element(patches)
            y_test_s.append(majority_element)

            patches = y_pred[i:i+5]
            majority_element = find_majority_element(patches)
            majority_pred.append(majority_element)

        # conpute indoor/outdoor classification accuracy score
        replacements = {
            "In_Arch": "In",
            "In_Constr": "In",
            "Forest": "Out",
            "Out_Constr": "Out",
            "Out_Urban": "Out"
        }

        IO_pred = [replacements.get(item, item) for item in majority_pred]
        IO_true = [replacements.get(item, item) for item in y_test_s]

        accuracy = accuracy_score(IO_true, IO_pred)
        accuracy_models.append(accuracy)
        print(f'Accuracy: {accuracy:.3f}')

        printAndSaveHeatmap(df,modelname,outputPath,use5Scentens)

    # Parameter Evaluation
    df_perf_acc = pd.DataFrame(columns=["Modelname"])

    if use5Scentens:
        csv_path_perforemance = outputPath / f'modelPerformance_5patches_5scentens.csv'
    else:
        csv_path_perforemance = outputPath/ f'modelPerformance_5patches.csv'
    
    accuracy_models = [ '%.3f' % elem for elem in accuracy_models ]
    df_perf_acc["Accuracy"] = accuracy_models

    # Throughput evaluation
    throughput_model =[]
    throughput = get_throughput_hailo(Dataset5Patch,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
    throughput_model.append(throughput)
    print(throughput_model)
    throughput_model = [ '%.2f' % elem for elem in throughput_model ]
    df_perf_acc["Throughput"] = throughput_model
    df_perf_acc.to_csv(csv_path_perforemance, index=False)

if __name__ == "__main__":
    main()