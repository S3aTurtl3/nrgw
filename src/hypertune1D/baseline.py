from scriptt import *

def get_plain_model_identifier(lr,
                        steps,
                        check_for_overfit_every,
                        desc):
   return "plain"+hashlib.md5(f"plainm{lr}{steps}_overfitCheck{check_for_overfit_every}_patience{PATIENCE_NUM_EPOCHS}_{desc}test".encode('utf-8')).hexdigest()

def get_plain_model_filename(lr,
                        steps,
                        check_for_overfit_every,
                        desc): 
   return get_plain_model_identifier(lr, steps, check_for_overfit_every, desc) + ".eqx"

def nll(model, data):
   model_output = jax.vmap(lambda ex: model.inference_without_vector_field_snapshots(ex))(data)
   return NLLLoss_2(model_output[COARSE_VAR_NAME], model_output[LOGP_NAME])


def train_plain_nnrg(model: WrapperForNNRGSubModule,
               dataloader: StreamingDataLoader,
               loss_key,
               lr: float,
               dataset_test,
               directory_model_saving,
               steps=10000,
               weight_decay=1e-5,
               print_every=100,
               check_for_overfit_every=100,
               desc="",
                save_every=1500):
  """Train a version of our architecture that lacks regularization """
  optim = optax.adamw(lr, weight_decay=weight_decay)
  opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
  wandb.init(
      project="plain-neural-renormalization-group", 
      config={
          "learning_rate": lr,
          "steps": steps,
          "weight_decay": weight_decay,
          "check_for_overfit_every": check_for_overfit_every,
          "description": desc,
          "lattice_size": dataloader.array.shape[1] if hasattr(dataloader, 'array') else 'N/A'
      }
  )

  tracker = OverfitTracker(patience=PATIENCE_NUM_EPOCHS*dataloader.array.shape[0]/dataloader.batch_size/check_for_overfit_every, min_delta=0.01)

  fname = get_plain_model_filename(lr, steps, check_for_overfit_every, desc)
  pth = os.path.join(directory_model_saving,fname)



  @eqx.filter_value_and_grad(has_aux=False)
  def loss(model, data, loss_key):

    model_output = jax.vmap(lambda data: model.inference_without_vector_field_snapshots(data))(data)


    main_loss = NLLLoss_2(model_output[COARSE_VAR_NAME], model_output[LOGP_NAME])
    total_loss = main_loss
    return total_loss

  @eqx.filter_jit
  def validation_loss(model, loss_key):
    model_output = jax.vmap(lambda ex: model.inference_without_vector_field_snapshots(ex))(dataset_test)
    val_loss = NLLLoss_2(model_output[COARSE_VAR_NAME], model_output[LOGP_NAME])
    return val_loss


  @eqx.filter_jit
  def make_step(model: WrapperForNNRGSubModule, opt_state, data, loss_key):

      value, grads = loss(model, data, loss_key)
      loss_key = jr.split(loss_key, 1)[0]
      updates, opt_state = optim.update(
          grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
      )
      model = eqx.apply_updates(model, updates)
      return value, model, opt_state, loss_key

  step = 0
  best_model = None
  best_loss = float('inf')
  loss_msgs = []
  loss_key, key_val = jr.split(loss_key, 2)
  overfitting = False
  while step < steps:
      val_loss = None
      start = time.time()

      data = dataloader(step)
      step = step + 1
      value, model, opt_state, loss_key = make_step(
          model, opt_state, data, loss_key
      )

      end = time.time()
      if (step % check_for_overfit_every == 0) or steps == steps-1:
        val_loss = validation_loss(model, key_val)
        key_val = jr.fold_in(key_val, step)
        tracker_verdict = tracker.update(val_loss)
        if tracker_verdict == "stop":
          overfitting = True
          print(f"Overfitting! val loss: {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
      if (step % print_every) == 0 or step == steps - 1:
          loss_msg = f"Step: {step}, Loss: {value}, Val loss: {val_loss}, Computation time: {end - start}"
          print(loss_msg)
          loss_msgs.append(loss_msg)
          computation_time = end-start
          wandb.log({
              "total_loss": float(value),
              "val_loss": float(val_loss) if val_loss is not None else None,
              "computation_time_per_step": computation_time
          }, step=step)
      if (step % save_every) == 0 or step == steps - 1 or step == 1:
        nrg_wrapper_saver(pth, {"depth": len(model.nnrg.submodules)}, best_model)
      if overfitting:
        break

  wandb.finish()
  if best_model is None:
      print("Best_model was None")
      best_model = model
  return best_model, (opt_state, loss_msgs)



def main():
    parser = argparse.ArgumentParser(description='Train baseline NNRG model.')
    parser.add_argument('--batch_size', type=int, required=True,help='Batch size')
    parser.add_argument('--seed', type=int, default=5678, help='Random seed') # TODO: remove this seed parameter because it doesn't do anything
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--lattice_size', required=True,type=int)
    parser.add_argument('--num_train_samples', required=True,type=int)
    parser.add_argument('--num_test_samples',required=True, type=int)
    parser.add_argument('--num_val_samples', required=True,type=int)
    parser.add_argument('--num_trials', required=True,type=int)
    parser.add_argument('--temp', required=True,type=float) # EFF: add burn in as a parameter else tune
    parser.add_argument('--out', required=True,help='should be a directory with longterm storage (so not local scratch)')
    parser.add_argument('--dir_model_weights', required=True, help='e.g. local scratch if it is not important to retain model weights')
    parser.add_argument('--check_overfit_every', type=int, default=100)

    
    args = parser.parse_args()

    LATTICE_SIZE_ISING = args.lattice_size
    # Setup keys
    key = jr.PRNGKey(5678)
    model_key, loader_key, loss_key, test_key, evaluation_key, key_vis = jr.split(key, 6)

    OUTPUT_DIR = args.out
    TEMP_DIR = args.dir_model_weights
    OUTPUT_FILE_NAME = "tuningPlain"+ hashlib.md5(f"tuningPlain{vars(args)}".encode('utf-8')).hexdigest() + ".json"
    temp_tag = f"{args.lattice_size}T{args.temp:g}".replace(".", "p")
    model_saving_dir = os.path.join(TEMP_DIR, "models", temp_tag)
    os.makedirs(model_saving_dir, exist_ok=True)
    OUTPUT_FILE_SUBDIR = os.path.join(OUTPUT_DIR, temp_tag)
    os.makedirs(OUTPUT_FILE_SUBDIR, exist_ok=True)
    OUTPUT_FILE_PTH = os.path.join(OUTPUT_FILE_SUBDIR, OUTPUT_FILE_NAME)
    PLACEHOLDER_ISING_MEAN = jnp.zeros(LATTICE_SIZE_ISING)
    PLACEHOLDER_ISING_STD = jnp.ones(LATTICE_SIZE_ISING)
    # Constants calculation
    OLD_BATCH_SIZE = 500


    INTEGRATED_TIME = None if args.temp == 0 else max(get_help_finding_int_time(test_key, args.temp, LATTICE_SIZE_ISING, 1.0, 1.0, n=100), LATTICE_SIZE_ISING)
    COEFF_FOR_BURN_IN= None if args.temp == 0 else 2
    BURN_IN = None if args.temp == 0 else COEFF_FOR_BURN_IN*INTEGRATED_TIME

    #TAINTED================
    # Generate a dataset of size 19000
    NUM_TRAIN_SAMPLES = args.num_train_samples
    NUM_SAMPLES_TEST = args.num_test_samples
    NUM_SAMPLES_VALIDATION = args.num_val_samples
    NUM_CHAINS = 100
    assert NUM_TRAIN_SAMPLES % NUM_CHAINS == 0 and NUM_SAMPLES_VALIDATION % NUM_CHAINS == 0 and NUM_SAMPLES_TEST % NUM_CHAINS == 0

    # Cache generated train/test/validation datasets to a file in the same directory
    # as the script's final output (OUTPUT_DIR), NOT the model-weights directory.
    # If that cache file already exists, load the datasets from it instead of
    # regenerating them.

    DATA_CACHE_FILE_NAME = "dataset_cache_" + hashlib.md5(
        f"data{LATTICE_SIZE_ISING}_{args.temp}_{NUM_TRAIN_SAMPLES}_{NUM_SAMPLES_TEST}_{NUM_SAMPLES_VALIDATION}_{args.seed}".encode('utf-8')
    ).hexdigest() + ".npz"
    DATA_CACHE_PATH = os.path.join(OUTPUT_DIR, DATA_CACHE_FILE_NAME)

    if os.path.exists(DATA_CACHE_PATH):
        cached = np.load(DATA_CACHE_PATH)
        full_dataset = jnp.array(cached["full_dataset"])
        dataset_mean = jnp.array(cached["dataset_mean"])
        dataset_std = jnp.array(cached["dataset_std"])
        test_dataset = jnp.array(cached["test_dataset"])
        validation_dataset = jnp.array(cached["validation_dataset"])
        loader_key = jr.fold_in(loader_key, 0)  # keep RNG stream consistent w/ non-cached path
    else:
        dataset_key, test_key_new, loader_key = jr.split(loader_key, 3)

        all_data = sample_from_continuous_relaxation_1D(dataset_key, NUM_TRAIN_SAMPLES + NUM_SAMPLES_TEST + NUM_SAMPLES_VALIDATION, LATTICE_SIZE_ISING, args.temp, INTEGRATED_TIME, BURN_IN, NUM_CHAINS)

        
        # Standardize the dataset
        all_data, dataset_mean, dataset_std = create_standardized_dataset(all_data)
        full_dataset = all_data[:NUM_TRAIN_SAMPLES]
        test_dataset = all_data[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST]
        validation_dataset = all_data[NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST:NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST+NUM_SAMPLES_VALIDATION]


        np.savez(
            DATA_CACHE_PATH,
            full_dataset=np.asarray(full_dataset),
            dataset_mean=np.asarray(dataset_mean),
            dataset_std=np.asarray(dataset_std),
            test_dataset=np.asarray(test_dataset),
            validation_dataset=np.asarray(validation_dataset),
        )

    # Instantiate the regular DataLoader
    dataloader = DataLoader(full_dataset, NNRGIsingConfig.BATCH_SIZE, loader_key)
    #=========================


    client = Client()

    LR_PARAM_NAME = "learning rate"
    NLL_METRIC_NAME = "NLL"
    MMD_METRIC_NAME = "MMD"
    

    def get_description_of_plain_job():
        return str(args) + f"lsize{LATTICE_SIZE_ISING}"
    
    def make_and_save_visualizations_of_best_models_plain(frontier, key_frontier, test_dataset):
        
        key_frontier_visualizations = jr.split(key_frontier, len(frontier))
        NUM_SAMPLES_BASIC_EVAL = 500
        comparison_dataset = test_dataset[:NUM_SAMPLES_BASIC_EVAL] * dataset_std + dataset_mean
        for i, (parameters, metrics, trial_index, arm_name) in enumerate(frontier):
            # visualize model samples compared to test dataset
            key_current_parameterization = key_frontier_visualizations[i]
            key_discrete_model, key_discrete_test = jr.split(key_current_parameterization)
            name_of_model = get_plain_model_filename(parameters[LR_PARAM_NAME], args.steps, args.check_overfit_every, get_description_of_plain_job()) # LEFT OFF
            nrg_model = load_model(os.path.join(model_saving_dir, name_of_model), WrapperForNNRG)
            configs_sampled_from_model = get_discrete_samples_from_model(nrg_model, dataset_mean, dataset_std, key_discrete_model, LATTICE_SIZE_ISING, NUM_SAMPLES_BASIC_EVAL)
            configs_from_test_dataset = get_discrete_samples(comparison_dataset, key_discrete_test)
            fig, stats = compare_model_vs_validation(configs_sampled_from_model, configs_from_test_dataset, n_show=10)
            fname = "OutputVis" + get_plain_model_identifier(parameters[LR_PARAM_NAME], args.steps, args.check_overfit_every, get_description_of_plain_job()) + ".pdf"
            fig.savefig(os.path.join(OUTPUT_FILE_SUBDIR, fname))

  
    client.configure_experiment(parameters=[

        ChoiceParameterConfig(
                name="learning rate",
                parameter_type="float",
                values=[
                    1e-3/4,
                    1e-3/4*2,
                    1e-3/4*2**2,
                    1e-3/4*2**3,

                ],
                is_ordered=True,
            ),
    ])

    client.configure_optimization(
    objective=f"-{NLL_METRIC_NAME}, -{MMD_METRIC_NAME}",
    outcome_constraints=[f"{NLL_METRIC_NAME} <= 300", f"{MMD_METRIC_NAME} <= {0.04*compute_number_of_latent_vars_being_regularized(args.lattice_size)}"],
)

    
    

    NUM_SAMPLES_BASIC_EVAL = 1000
    for _ in range(args.num_trials):
        evaluation_key, key_nll, key_sample_quality, key_ot_penalty = jr.split(evaluation_key, 4)
        trials = client.get_next_trials(max_trials=1)
        per_trial_loss_msgs = []
        for trial_index, parameters in trials.items():

            # Model initialization
            nrg_model = WrapperForNNRG(depth=int(jnp.ceil(jnp.log2(LATTICE_SIZE_ISING))), key=model_key)


    
            nrg_model, (_, loss_msgs) = train_plain_nnrg(nrg_model,
               dataloader,
               loss_key,
               lr=parameters[LR_PARAM_NAME],
               dataset_test=validation_dataset,
               directory_model_saving=model_saving_dir,
               steps=args.steps,
               check_for_overfit_every=args.check_overfit_every,
               desc=get_description_of_plain_job())
            per_trial_loss_msgs.append(loss_msgs)
            inference_info = ModelInferenceInfo(nrg_model, PLACEHOLDER_ISING_MEAN, PLACEHOLDER_ISING_STD)
            sample_quality = evaluate_sample_quality_nnrg(inference_info, test_dataset, key_sample_quality, LATTICE_SIZE_ISING)
            test_loss = nll(nrg_model, test_dataset)
            raw_data = {NLL_METRIC_NAME: float(test_loss), MMD_METRIC_NAME: float(sample_quality)}

            client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    frontier = client.get_pareto_frontier()
    make_and_save_visualizations_of_best_models_plain(frontier, evaluation_key, test_dataset)
    with open(OUTPUT_FILE_PTH, "w") as file:
      json.dump({"frontier": make_json_serializable(frontier), "loss_msgs": per_trial_loss_msgs}, file)



if __name__ == '__main__':
    # To run in Colab without crashing on sys.argv:
    
    if 'ipykernel' in sys.modules:
        # rendering the following as a sys.arvlist: --batch_size=50 --steps=100 --temp=199 --num_trials=3 --num_train_samples=100 --num_test_samples=100 --num_val_samples=100 --lattice_size=32
        sys.argv = ['', '--batch_size=50', '--steps=100', '--temp=199', '--num_trials=3',  '--num_train_samples=100', '--num_test_samples=100', '--num_val_samples=100', '--lattice_size=32', '--out=/', '--dir_model_weights=/' ]
    main()
