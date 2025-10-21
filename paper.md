---
title: 'ctSpyderFields: A Python package for visual field reconstruction in spiders'
tags:
  - Python
  - Vision
  - Eye evolution
  - Computerized Tomography

authors:
  - name: Massimo De Agrò
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: CIMeC, University of Trento, Italy
   index: 1
 - name: Institution Name, Country
   index: 2
date: 21/10/2025
bibliography: paper.bib

---

# Statement of need

Visual field measurements in spiders have a long history, dating back nearly a century. Homann (1928) 
was the first to use the reflective properties of the tapetum and the natural image formation by the eye 
to deduce optical parameters such as the field of view, focal length, and receptor spacing. His technique
was based on direct optical observation of the retina in living specimens, using the eye's own optics to
visualize the retinal mosaic. Land (1969) adapted this standard microscope setup into an ophthalmoscope 
by adding an extra lens. This additional lens is positioned so that it images the back focal plane of 
the microscope’s objective onto the eyepiece. Because the back focal plane is conjugate with infinity, 
this setup ensures that the retinal image is viewed as though it were formed at infinity. 
This modification captures a sharp, angularly calibrated image of the retina. Both methods, 
however, depend on the presence of reflecting structures, namely the tapetum, and are thus limited to
spider eyes that possess these features. Stowasser et al., (2017) further refined this approach by 
making use of the autofluorescence of the photoreceptors themselves instead of tapetal reflection to
image the retina and thus broadening the applicability of ophthalmoscopic techniques to a wider range
of invertebrate eyes. Despite these advances, several key limitations remain. If a particular species,
developmental stage, or even certain regions of the retina produce only weak autofluorescence, image quality
and accuracy in determining refractive properties may suffer, potentially leading to misinterpretation of 
the refractive state. Moreover, as with all in vivo optical techniques, the method requires the animal to 
be still, and any residual movement in the live animal may blur the autofluorescent signal or alter the 
observed focal relationships.
In recent decades, micro-computed tomography (micro-CT) has emerged as a powerful tool for high-resolution,
three-dimensional visualization of animal tissues (see du Plessis et al., 2017 for an overview). In contrast
to the ophthalmoscopic methods, micro-CT does not rely on the optical properties such as autofluorescence or
the refractive index differences, making it broadly applicable across a wide range of tissues and organisms,
enabling standardized imaging protocols across different samples. Volumetric data from micro-CT can be re-sliced
and examined from multiple angles, offering insights into the full anatomical context of the structures.
Finally, micro-CT scanning does not need live specimens and can be readily performed on museum specimens,
further extending the utility of this method. Thus, micro-CT methods have a huge potential to complement 
micro-ophthalmoscopic studies.

# Summary



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements


# References