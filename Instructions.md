# Coursework

This repository is for the submission of your coursework for the S1 Principles of Data Science module. 

You have been given access with the role of "Maintainer" for this repository which will expire on the 17th of December at midnight, which is the submission deadline for this work.

You should use this repository to submit **both** your code **and** your report.

## The problem

The problem itself has been released on the course [Moodle page](https://www.vle.cam.ac.uk/course/view.php?id=252189#section-2) and also in the course [Gitlab project](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_principles_of_data_science).

The code should be simple to use, written in Python (or Python utilising user created C/Fortran libraries with python bindings), and be run-able from the command line, i.e.:

```bash
$ python src/solve_part_e.py
```

## Criteria

The goal of this coursework project is for you to demonstrate that you can apply some of the statistical analysis techniques we have discussed in the course.

You will need to demonstrate the following:

 - You can write clear, readable code that is simple to run
 - You can justify, from a statistical perspective, the approach you have taken to solve the problem

You will be assessed on:

 - The clarity and ease with which your code can be read and run
 - The scientific justifcation in your report write-up
 - How informative and well-explained your plots are
 - The writing quality and style including the spelling and grammar

For reference my solutions to parts (f) and (g), which are not particularly optimised, run on my local machine in about an hour each, using the `conda` environment I have used for all of my lectures.

The project should contain a `README.md` file which describes **exactly** how to run the code. You should provide a Docker container or a Conda environment so that I can run the code easily. Please include your `README.md` file as an Appendix of your report (that will not count towards the word limit) and provide some details of how long it took you to run your code and the specifications of the machine you ran it on.
You should ensure your code is well documented with regular comments which explain what your code is doing.


Please remember the relevant section of the handbook that refers to use of ChatGPT in your academic work, specifically the following passage:

"Generation tools must be used transparently:

All use of auto-generation tools must be explicitly cited in every instance of their use.
This applies to generating code, whether used for prototyping, creation, reformatting, or any other purpose. Students should add the citations to the README in home repository, and in any accompanying reports stating the prompts submitted, where the output was used, and how it was modified.
When used in conjunction with submitted reports for drafting, proofreading, suggesting alternative wordings, or for any other task it should be explicitly noted in an appendix to the report with the prompts submitted, where the output was used, and how it was modified.
Failure to adequately cite use of these tools is considered academic misconduct. "

## Submission

You should write a report of **no more than 3000 words** to accompany the software you write to solve the problem.
The report should be written in LaTeX.
Your report should contain two sections, labelled "Section A" and "Section B".

Section A should answer parts (a) - (e) with concise answers and short mathematical proofs.
It should be no longer than 500 - 1000 words.

Section B should answer parts (f) and (g) and contain a more detailed description and justification of your method.
You may wish to further partition this into subsections, for example

- Introduction
- Methodology
- Analysis
- Discussion 

There are several ways to solve this problem and produce a sensible result. 
It is more important that you explain and justify your approach, and submit suitable code to perform the task, than it is to get the answer "right".

Both your code **and** your report should be submitted in this repository. 
The generated PDF of your report should be placed in the `report` subdirectory.
The `README.md` file should contain short and simple instructions on how to run your code.
Please **make sure** your code is portable and reproducible by utilising environment and containerisation tools like Conda and Docker. 
The code should be runnable in the container without any effort beyond generating the image.
I will not attempt to debug your code if it does not run first time.  
