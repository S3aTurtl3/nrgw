import diffrax

NUM_STEPS = 10
def differential_equation_solve(terms, solver, t0, t1, dt0, y0, args):
    return diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, args)

def differential_equation_solve_with_saveat(terms, solver, t0, t1, dt0, y0, args, saveat):
    return diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, args, saveat=saveat)



def _diffeqsolve_with_fixed_num_steps(num_steps, terms, solver, t0, t1, y0, args):
    dt0 = (t1-t0) / num_steps
    return diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, args, stepsize_controller=diffrax.ConstantStepSize())
def _diffeqsolve_with_fixed_num_steps_with_saveat(num_steps, terms, solver, t0, t1, y0, args, saveat):
    dt0 = (t1-t0) / num_steps
    return diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, args, stepsize_controller=diffrax.ConstantStepSize(), saveat=saveat)

