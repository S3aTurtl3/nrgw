import jax
from scriptt import WrapperForNNRG, NLLLoss_2, COARSE_VAR_NAME, LOGP_NAME

def just_nll(model: WrapperForNNRG, data):
   model_output = jax.vmap(lambda ex: model.inference_without_vector_field_snapshots(ex))(data)
   return NLLLoss_2(model_output[COARSE_VAR_NAME], model_output[LOGP_NAME])