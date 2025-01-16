import os
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from pathlib import Path
from utils import printAndSaveClassReport,printClassificationReport,get_max_class_with_threshold,get_trueClass,find_majority_element,printAndSaveHeatmap
from hailoEval_utils import get_pred_hailo,get_throughput_hailo, HailoCLIPImage, HailoCLIPText,getModelfiles,get_throughput_hailo_image,HailofastCLIPImage

"""
Evaluates CLIP and TinyCLIP models on Hailo yy.
Same code as on PC
"""

# Pathes
outputPath = Path("Evaluation/Data/HailoCutThused")
Dataset5Patch224px = Path("candolle_5patches_224px")
Dataset5Patch = Path("candolle_5patch")
tinyClipModels = Path("tinyClipModels")
models_path = "Evaluation/modelscut"

def main():
    InOutTh_dict = {'RN50':0.63, 'RN50x4':0.52, 'TinyClip19M':0.51, 'RN101':0.78, 'TinyClip30M':0.72,'TinyClip19M16Bit':0.45}
    # InOutTh_dict = {'RN50':0.5, 'RN50x4':0.5, 'TinyClip19M':0.5, 'RN101':0.5, 'TinyClip30M':0.5, 'TinyClip19M16Bit':0.5}
    df_perf_acc = pd.DataFrame(columns=["Modelname"])
    accuracy_models = []
    balanced_accuracy_models = []
    useNewCut = False
    if "cut" in models_path:
        useNewCut = True
        
    # Throughput evaluation
    throughput_model_mean = []
    throughput_model_std = []
    throughput_model_mean_image = []
    throughput_model_std_image = []
    
    models_list = next(os.walk(models_path), (None, [], None))[1]
    # Only TinyClip
    # models_list = [model for model in models_list if "Tiny" in model]
    for model_folder in models_list:
        folder_path = models_path + "/" + model_folder
        gemm_path, hef_path, textemb_path, textemb5S_path, OnnxPostp_path = getModelfiles(folder_path)
        
        # set text embeddings and post process
        textEmb = textemb5S_path
        postP = OnnxPostp_path
        modelname = model_folder
        print(f"Model name:{modelname}")
        print(f"Postprocess path:{postP}")
        print(f"Emb path:{textEmb}")
        
        hailoImagemodel = HailofastCLIPImage(hef_path, postP,isnewCut=useNewCut)
        # hailoImagemodel = HailofastCLIPImage(hef_path, gemm_path)
        hailoTextmodel = HailoCLIPText(textEmb)
        
        
        use5Scentens = hailoTextmodel.getuse5Scentens()

        # Path to csv
        if use5Scentens:
            csv_path_predictions = outputPath / f'pred_{modelname}_5patches_5scentens.csv'
        else:
            csv_path_predictions = outputPath / f'pred_{modelname}_5patches.csv'

        # check if csv already exists
        if os.path.exists(csv_path_predictions):
            df_5patch = pd.read_csv(csv_path_predictions)
            df_5patch = df_5patch.sort_values(["Scene","Image"],ascending=[True,True])
            df_5patch = get_trueClass(df_5patch)
        else:
            df_pred = get_pred_hailo(Dataset5Patch,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
            df_pred.to_csv(csv_path_predictions, index=False)
            df_5patch = pd.read_csv(csv_path_predictions)
            df_5patch = df_5patch.sort_values(["Scene","Image"],ascending=[True,True])
            df_5patch = get_trueClass(df_5patch)
        df = df_5patch.copy()
        threshold = InOutTh_dict[modelname]
        df['y_predIO'] = df.apply(get_max_class_with_threshold, axis=1, threshold=threshold)

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

        
        # balanced_accuracy = balanced_accuracy_score(IO_true, IO_pred)
        # accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test_s, majority_pred)
        accuracy = accuracy_score(y_test_s, majority_pred)
        accuracy_models.append(accuracy)
        balanced_accuracy_models.append(balanced_accuracy)
        print(f'Accuracy: {accuracy:.3f}')
        print(f'Balanced accuracy: {balanced_accuracy:.3f}')

        # Heatmap
        printAndSaveHeatmap(df,modelname,outputPath,use5Scentens)
        
        # Classification Report (Bar plot)
        classificationReport = printClassificationReport(df, modelname)
        del classificationReport['accuracy']
        df_classReport = pd.DataFrame.from_dict(classificationReport,orient='index')
        printAndSaveClassReport(classificationReport,modelname,outputPath)
        
        # Path to csv
        if use5Scentens:
            csv_path_classReport = outputPath / f'classReport_{modelname}_5patches_5scentens.csv'
        else:
            csv_path_classReport = outputPath / f'classReport_{modelname}_5patches.csv'
            
        df_classReport.to_csv(csv_path_classReport, index=True)
        
        throughput_mean,throughput_std = get_throughput_hailo(Dataset5Patch,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
        throughput_model_mean.append(throughput_mean)
        throughput_model_std.append(throughput_std)
        print(f"\nThrouputs: {throughput_model_mean}")
        
        throughput_mean_image,throughput_std_image = get_throughput_hailo_image(Dataset5Patch,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
        
        # Free device as early as possible
        del hailoImagemodel
        throughput_model_mean_image.append(throughput_mean_image)
        throughput_model_std_image.append(throughput_std_image)
        print(f"\nThrouputs Image: {throughput_model_mean_image}")
        
        
        
        df_perf_acc = df_perf_acc._append({
            "Modelname": modelname,
        }, ignore_index=True)
        
    if use5Scentens:
        csv_path_perforemance = outputPath / f'modelPerformance_5patches_5scentens.csv'
    else:
        csv_path_perforemance = outputPath / f'modelPerformance_5patches.csv'
    
    throughput_model_mean = ['%.2f' % elem for elem in throughput_model_mean]
    throughput_model_mean_image = ['%.2f' % elem for elem in throughput_model_mean_image]
    
    accuracy_models = [ '%.3f' % elem for elem in accuracy_models ]
    balanced_accuracy_models = [ '%.3f' % elem for elem in balanced_accuracy_models ]
    df_perf_acc["Accuracy"] = accuracy_models
    df_perf_acc["Balanced accuracy"] = balanced_accuracy_models
    df_perf_acc["Throughput (it/s)"] = throughput_model_mean
    df_perf_acc["Throughput Image (it/s)"] = throughput_model_mean_image
    df_perf_acc["Throughput std"] = throughput_model_std
    df_perf_acc["Throughput Image std"] = throughput_model_std_image
    df_perf_acc.to_csv(csv_path_perforemance, index=False)

if __name__ == "__main__":
    main()