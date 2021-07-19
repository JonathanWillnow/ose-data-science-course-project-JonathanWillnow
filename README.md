# OSE data science project summerterm 2021 by Jonathan Willnow

This is the final project (in progress) of the OSE data science course summerterm 2021 by Jonathan Willnow. 

## Project overview

This project replicates and comments on the findings of 
> [Dreher et. al (2021): Aid, China, and Growth: Evidence from a New Global Development Finance Dataset. American Economic Journal: Economic Policy, vol. 13(2), May 2021 (pp. 135-174).](https://www.aeaweb.org/articles?id=10.1257/pol.20180631)

The Belt and Road Initiative (BRI), better known as the "new silk road initiative" is just one of many instances of China´s overseas activities of financing development, especially known to the Europeans. Other projects, mostly infrastructure projects (by transaction value), link China within Asia and with the African continent. This role of China as significant donor raises strong opinions, but the debate was based on only little facts since most of the details are not officially reported. The paper at hand uses the Tracking Underreported Financial Flows (TUFF) methodology to introduce a new dataset that provides the needed evidentiary foundations that was needed for this issue.

I will focus on the following questions: 

* What determines the allocation of Chinese development finance?
* Does Chinas financial development finance lead to economic growth?

To answer this question on whether and how Chinese development finance affects economic growth, instrumental variables are employed that make use of the year-to-year changes in the supply of Chinese development finance (which will be introduced) in tandem with cross-sectional variation which is determined by the probability that countries receive such funding. Additional, independent extensions from the work of Dreher et al. (2021) will be added.



<a href="https://nbviewer.jupyter.org/github/OpenSourceEconomics/ose-data-science-course-project-JonathanWillnow/blob/master/JonathanWillnowOSE.ipynb"
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">
</a>
<a href="https://mybinder.org/v2/gh/OpenSourceEconomics/ose-data-science-course-project-JonathanWillnow/master?filepath=JonathanWillnowOSE.ipynb"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>

## Reproducibility


![Continuous Integration](https://github.com/OpenSourceEconomics/ose-template-course-project/workflows/Continuous%20Integration/badge.svg)

For full reproducibility of this project, a continuous integration workflow was set up using [GitHub Actions CI](https://docs.github.com/en/actions). I also provided an environment.yml file of my environment to ensure full reproducibility of my notebook.


## Acknowledgement

This replication was only possible since Dreher et al. (2021) provide their data and their STATA code. I would also like to thank all the awesome people that provide their knowledge by creating and managing the various documentations and coding examples available all over the internet. Special thanks to Prof. Dr. Philipp Eisenhauer and Prof. Dr. Dominik Liebl, who have taught me a lot in the last two semesters and without whom I would not have been able to complete this project.


## Sources


* Borenstein et al. (2009): Introduction to Meta-Analysis. Michael Borenstein, John Wiley & Sons, 2009


* Dreher, Axel, Andreas Fuchs, Bradley Parks, Austin Strange, and Michael J. Tierney. 2021. "Aid, China, and Growth: Evidence from a New Global Development Finance Dataset." American Economic Journal: Economic Policy, 13 (2): 135-174.


* Labrecque, J., & Swanson, S. A. (2018). Understanding the Assumptions Underlying Instrumental Variable Analyses: a Brief Review of Falsification Strategies and Related Tools. Current epidemiology reports, 5(3), 214–220. https://doi.org/10.1007/s40471-018-0152-1


* publichealth.columbia.edu (2021): Difference-in-Difference Estimation, online source [https://www.publichealth.columbia.edu/research/population-health-methods/difference-difference-estimation], last access 04.07.2021


* Zhou et al. (2016). Difference-in-Differences Method in Comparative Effectiveness Research: Utility with Unbalanced Groups. Applied health economics and health policy, 14(4), 419–429. https://doi.org/10.1007/s40258-016-0249-y


* Strange et al. (2017): AidData's Tracking Underreported Financial Flows (TUFF) Methodology, Version 1.3. Williamsburg, VA: AidData at William & Mary.




