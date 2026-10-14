# Running the research exports

The numbered research scripts are notebook exports and may rely on state created by previous steps, datasets, trained checkpoints, and Colab-specific services. They are not independent production entry points. Install the dependencies from the preserved notebook environment command comments in an appropriate isolated environment before executing the relevant cells; Python scripts no longer contain IPython shell syntax or automatically run system package installers.

CI checks Python compilation without importing research dependencies or training models. Passing compilation does not validate reported metrics, reproduce experiments, or imply that the ablation study has been run. The SCORE-BN ablation section remains an experiment plan until actual runs and evidence are supplied. Review file paths and artifact availability before execution.
