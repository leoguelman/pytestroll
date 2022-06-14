import pandas as pd

def stan_model_summary(stan_fit):
    """
    Parameters
    ----------
    stan_fit : A Stan fit object
    Returns
    -------
    A pandas data frame with summary of posterior parameters.
    """
    summary_dict = stan_fit.summary()
    summary_df = pd.DataFrame(summary_dict['summary'],
                  columns=summary_dict['summary_colnames'],
                  index=summary_dict['summary_rownames'])
    return summary_df