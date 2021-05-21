# OSE data science project summerterm 2021

This is the final project of the OSE data science course summerterm 2021 by Jonathan Willnow. 

## Project overview

This project replicates and comments on the findings of Dreher et. al (2021): Aid, China, and Growth: Evidence from a New Global Development Finance Dataset. American Economic Journal: Economic Policy, vol. 13(2), may 2021 (pp. 135-74).

* Please ensure that a brief description of your project is included in the [README.md](https://github.com/HumanCapitalAnalysis/template-course-project/blob/master/README.md), which provides a proper citation of your baseline article. Also, please set up the following badges that allow to easily access your project notebook.

<a href="https://nbviewer.jupyter.org/github/OpenSourceEconomics/ose-template-course-project/blob/master/example_project.ipynb"
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">
</a>
<a href="https://mybinder.org/v2/gh/OpenSourceEconomics/ose-template-course-project/master?filepath=example_project.ipynb"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>

## Reproducibility


![Continuous Integration](https://github.com/OpenSourceEconomics/ose-template-course-project/workflows/Continuous%20Integration/badge.svg)

For full reproducibility of this project, an continuous integration workflow was set up using [GitHub Actions CI](https://docs.github.com/en/actions). I also provided an environment.yml file of my environment to ensure full reproducibility of my notebook.

In some cases you might not be able to run parts of your code on  [GitHub Actions CI](https://docs.github.com/en/actions) as, for example, the computation of results takes multiple hours. In those cases you can add the result in a file to your repository and load it in the notebook. See below for an example code.

```python
# If we are running on GitHub Actions CI we will simply load a file with existing results.
if os.environ.get("CI") == "true":
  rslt = pkl.load(open('stored_results.pkl', 'br'))
else:
  rslt = compute_results()

# Now we are ready for further processing.
...
```

However, if you decide to do so, please be sure to provide an explanation in your notebook explaining why exactly this is required in your case.

## Structure of notebook

A typical project notebook has the following structure:

* presentation of baseline article with proper citation and brief summary

* using causal graphs to illustrate the authors' identification strategy

* replication of selected key results

* critical assessment of quality

* independent contribution, e.g. additional external evidence, robustness checks, visualization

There might be good reason to deviate from this structure. If so, please simply document your reasoning and go ahead. Please use the opportunity to review other student projects for some inspirations as well.
