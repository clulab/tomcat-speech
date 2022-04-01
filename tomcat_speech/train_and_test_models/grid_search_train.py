# create a grid search and run train_multitask.py for each set of parameters

from copy import deepcopy
import sys

single_task = False
if len(sys.argv) > 1:
    if sys.argv[1].lower() == "singletask" or sys.argv[1].lower() == "single task" or sys.argv[1].lower() == "single":
        from tomcat_speech.train_and_test_models.train_single_task import *
        import tomcat_speech.models.parameters.singletask_gridsearch_config as config
        print("Using single task grid search")
        single_task = True
    else:
        from tomcat_speech.train_and_test_models.train_multitask import *
        import tomcat_speech.models.parameters.multitask_gridsearch_config as config
        print("sys.argv[1] is not SINGLE; using multitask grid search")
else:
    # default to multitask
    from tomcat_speech.train_and_test_models.train_multitask import *
    import tomcat_speech.models.parameters.multitask_gridsearch_config as config
    print("No config type specified; defaulting to multitask grid search")

# get the params
params = config.model_params

# set the random seeds and get device
device = set_cuda_and_seeds(config)

# get the data
if not config.model_params.use_distilbert:
    data, loss_fx, sampler, num_embeddings, pretrained_embeddings = load_data(device, config)
else:
    data, loss_fx, sampler = load_data(device, config)
    num_embeddings = None
    pretrained_embeddings = None

all_best_f1s = []

# this assumes that the parameters included in the grid search are
#   LR
#   Dropout
#   Final hidden dim
#   batch size
#   to add/remove, alter this nested for loop
for l_rate in params.lr:
    for dpt in params.dropout:
        for fhid in params.final_hidden_dim:
            for b_size in params.batch_size:

                # create save location
                output_path = os.path.join(
                    config.exp_save_path,
                    str(config.EXPERIMENT_ID)
                    + "_"
                    + config.EXPERIMENT_DESCRIPTION
                    + str(date.today()),
                )

                # make sure the full save path exists; if not, create it
                os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(output_path))

                # copy the config file into the experiment directory
                shutil.copyfile(config.CONFIG_FILE, os.path.join(output_path, "config.py"))

                # add stdout to a log file
                with open(os.path.join(output_path, "log"), "a") as f:
                    if not config.DEBUG:
                        sys.stdout = f

                        this_model_params = deepcopy(config.model_params)
                        this_model_params.lr = l_rate
                        this_model_params.dropout = dpt
                        this_model_params.final_hidden_dim = fhid
                        this_model_params.batch_size = b_size
                        print(this_model_params)

                        if not single_task:
                            best_f1s = train_multitask(data, loss_fx, sampler, device, output_path, config, num_embeddings, pretrained_embeddings, extra_params=this_model_params)
                        else:
                            best_f1s = train_single_task(data, loss_fx, sampler, device, output_path, config, num_embeddings, pretrained_embeddings, extra_params=this_model_params)

                        all_best_f1s.append(best_f1s)

with open(f"{output_path}/grid_search_results.csv", 'w') as wf:
    for item in all_best_f1s:
        wf.write(f'{",".join(item)}\n')

