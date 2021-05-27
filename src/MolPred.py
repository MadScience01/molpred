import pandas as pd
import argparse
import numpy as np
from multiprocessing import Pool
import glob2
import multiprocessing

from sklearn.preprocessing import MinMaxScaler
from functools import partial

import os
from datetime import datetime
import joblib


def transform_freqs_to_colnames(input_transitions):
    input_transitions = input_transitions.T
    new_header = input_transitions.iloc[0]  # grab the first row for the header
    input_transitions = input_transitions[1:]  # take the data less the header row
    input_transitions.columns = new_header  # set the header row as the df header
    return input_transitions


def find_closest_freq(transition, spec):
    df_sort = spec.iloc[(spec['Frequency'] - transition).abs().argsort()[:1]]
    return (df_sort['Frequency'])


def get_model_with_lowest_mae(molecule):
    error_tables_dir = '..' + os.sep + 'error_tables'
    history_files = glob2.glob(error_tables_dir + os.sep + molecule + '*Epochs' + os.sep + '*False.csv')
    df_mae = pd.DataFrame(columns=('model_file', 'min_mae'))
    for history_file in history_files:
        df_tmp = pd.DataFrame(columns=('model_file', 'min_mae'))
        history_df = pd.read_csv(history_file)
        min_mse = history_df.min()[1]
        model_file = history_file.replace('error_tables', 'models').replace('.csv', '.h5')
        df_tmp.loc[0] = [model_file, min_mse]
        df_mae = df_mae.append(df_tmp)
    df_mae = df_mae.sort_values(by='min_mae')
    best_result = df_mae.iloc[0]
    return best_result


def get_prediction(input_object, initial_spectrum):
    from tensorflow.keras.models import load_model
    import tensorflow as tf

    input_transitions = input_object[0]
    molecule = input_object[1]
    model_file = input_object[2][0]
    label_scaler_object = input_object[3]
    dataset_scaler_object = input_object[4]
    model_min_mae = input_object[5]
    input_intensities = transform_freqs_to_colnames(input_transitions)
    scaled_input_type1 = dataset_scaler_object.transform(input_intensities)

    # ------------------------------
    # this block enables GPU enabled multiprocessing
    #core_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=5)
    core_config = tf.compat.v1.ConfigProto()
    core_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=core_config)
    tf.compat.v1.keras.backend.set_session(session)
    # -------------------------------
    model = load_model(model_file, compile=False)
    predictions = model.predict(scaled_input_type1, batch_size=1)
    descaled_predictions = label_scaler_object.inverse_transform(predictions)
    return [molecule, model_file, predictions, descaled_predictions, model_min_mae]


def generate_single_input(initial_spectrum, molecule, label_scalers, dataset_scalers):
    splat_path = '..' + os.sep + 'splatalogue'
    models_dir = '..' + os.sep + 'models'
    splat_transitions = np.loadtxt(splat_path + os.sep + molecule + '.txt', comments="//")
    spectrum = pd.DataFrame(columns=('Frequency', 'Intensity'))
    for splat_transition in splat_transitions:
        # splatalogue transitions and example ALCHEMI data come in GHz, so no need to convert.
        closest_freq = find_closest_freq(splat_transition, initial_spectrum)
        # check if closest frequency is within 10MHz. otherwise, assign intensity zero.
        if abs(closest_freq.values[0] - splat_transition) < 0.01:
            value_to_append = initial_spectrum.iloc[closest_freq.index]
            value_to_append['Frequency'] = splat_transition
            spectrum = spectrum.append(value_to_append, ignore_index=True)
        else:
            df = pd.DataFrame(columns=('Frequency', 'Intensity'))
            df.loc[0] = [splat_transition, 0.0]
            spectrum = spectrum.append(df, ignore_index=True)
            print("For transition: " + str(splat_transition) + ", closest Frequency is: " + str(
                closest_freq.values[
                    0]) + " which does not seem match with current transition, setting transition to zero")
    model_file = get_model_with_lowest_mae(molecule)
    model_dir = model_file[0][1:model_file[0].find('NOISE') + 5]
    model_dir = model_dir[model_dir.rfind(os.sep) + 1:]
    label_scaler_object = label_scalers[model_dir]
    dataset_scaler_object = dataset_scalers[model_dir]
    model_min_mae = model_file[1]
    return [spectrum, molecule, model_file, label_scaler_object, dataset_scaler_object, model_min_mae]


def generate_inputs(initial_spectrum, molecule_list, label_scalers, dataset_scalers):
    input_objects = []
    for molecule in molecule_list:
        input_object = generate_single_input(initial_spectrum, molecule, label_scalers, dataset_scalers)
        input_objects.append(input_object)
    return input_objects


def get_scalers():
    label_scalers_dict = {}
    dataset_scalers_dict = {}
    scaler_files = glob2.glob('..' + os.sep + 'scalers' + os.sep + '*')
    for scaler_file in scaler_files:
        scaler = joblib.load(scaler_file)
        dict_key = scaler_file[scaler_file.rfind(os.sep) + 1:scaler_file.rfind('.')]
        if scaler_file.__contains__('label'):
            label_scalers_dict[dict_key] = scaler
        else:
            dataset_scalers_dict[dict_key] = scaler
    return label_scalers_dict, dataset_scalers_dict


def save_scalers(training_dataset_dir):
    training_dataset_files = glob2.glob(training_dataset_dir + os.sep + '*train.csv')
    for training_dataset_file in training_dataset_files:
        dataset = pd.read_csv(training_dataset_file)
        dataset = dataset.drop(['spectra_id', 'Unnamed: 0'], axis=1)
        labels = dataset[dataset.columns[-2:]]
        dataset = dataset[dataset.columns[:-2]]
        dataset_scaler = MinMaxScaler().fit(dataset)
        labels_scaler = MinMaxScaler().fit(labels)
        dict_key = training_dataset_file[training_dataset_file.rfind(os.sep) + 1:-10]
        joblib.dump(dataset_scaler, '..' + os.sep + 'scalers' + os.sep + dict_key + '.dataset_scaler')
        joblib.dump(labels_scaler, '..' + os.sep + 'scalers' + os.sep + dict_key + '.labels_scaler')


def get_spectrum_scaler_object(initial_spectrum):
    scaler = MinMaxScaler()
    spectrum_scaler = scaler.fit(initial_spectrum)
    return spectrum_scaler

def run_program(times_to_repeat):
        execution_times = []
        for i in range(0,times_to_repeat):
            parser = argparse.ArgumentParser()
            parser.add_argument('--specfile', dest='specfile', help="file containing the spectrum to be analyzed.",
                                required=False)
            cs = parser.parse_args()

            specfile = cs.specfile

            # Config Section:
            datasets_dir = '..' + os.sep + 'datasets'
            molecules = ["CO", "HCO+", "SiO", "CH3CN"]
            # End Config Section

            # Pre Processing Section
            df_initial_spectrum = pd.read_csv(specfile, sep='\t')
            df_initial_spectrum.columns = ['Frequency', 'Intensity']
            df_initial_spectrum['Intensity'] = df_initial_spectrum['Intensity'].fillna(0.0)
            # save_scalers(datasets_dir)
            label_scalers, dataset_scalers = get_scalers()
            df_errors = pd.DataFrame(
                columns=('TestedMolecule', 'predicted_log(N)', 'predicted_Tex', 'descaled_predicted_log(N)',
                         'descaled_predicted_Tex', 'min_mae', 'model_file'))
            input_objects = generate_inputs(df_initial_spectrum, molecules, label_scalers, dataset_scalers)

            # End of Preprocessing

            # Prediction Section
            partial_func = partial(get_prediction, initial_spectrum=df_initial_spectrum)
            # workers = int(mp.cpu_count()/2)
            multiprocessing.set_start_method('spawn', force=True)
            workers = 1
            pool = Pool(processes=workers)
            # pool.map(f, range(mp.cpu_count()))
            start = datetime.now()
            results = pool.map(partial_func, input_objects)
            pool.close()
            pool.join()
            end = datetime.now()
            elapsed_time = end - start

            ## End of Prediction

            ### Post Processing Section.
            print("Prediction Results:")
            print("")
            for result in results:
                molecule = result[0]
                model_file = result[1]
                scaled_predicted_T = result[2][0][0]
                scaled_predicted_N = result[2][0][1]
                descaled_predicted_T = result[3][0][0]
                descaled_predicted_N = result[3][0][1]
                min_mae = result[4]
                df_error = pd.DataFrame(columns=(
                    'TestedMolecule', 'predicted_log(N)', 'predicted_Tex', 'descaled_predicted_log(N)',
                    'descaled_predicted_Tex', 'min_mae', 'model_file'))
                df_error.loc[0] = [molecule, scaled_predicted_N, scaled_predicted_T, descaled_predicted_N,
                                   descaled_predicted_T, min_mae, model_file]
                df_errors = df_errors.append(df_error, ignore_index=True)
                print("---------------------")
                print("Molecule: " + molecule)
                print("Model File: " + model_file)
                print("Scaled Predicted Tex: " + str(scaled_predicted_T))
                print("Scaled Predicted log(N): " + str(scaled_predicted_N))
                print("Descaled Predicted Tex: " + str(descaled_predicted_T))
                print("Descaled Predicted log(N): " + str(descaled_predicted_N))
                print("min_mae: " + str(min_mae))
                print("---------------------")
            print("Predictions took: " + str(elapsed_time.total_seconds() * 1000) + " milliseconds using " + str(
                workers) + " workers")
            prediction_savefile = '..' + os.sep + 'predictions' + os.sep + str(
                datetime.now().strftime('%Y_%m_%d_%H_%M')) + '_molpred_predictions.csv'
            #df_errors.to_csv(prediction_savefile)
            #print('Predictions saved to file: ' + prediction_savefile)
            execution_times.append(elapsed_time.total_seconds())
        print("average execution time in "+str(times_to_repeat)+" executions: "+str(np.average(execution_times))+" seconds")
        print('Program Finished.')

#set to -1 to disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    run_program(1)



