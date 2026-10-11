## Context
In astronomy, stellar classification is the classification of stars based on their spectral characteristics. The classification scheme of galaxies, quasars, and stars is one of the most fundamental in astronomy. The early cataloguing of stars and their distribution in the sky has led to the understanding that they make up our own galaxy and, following the distinction that Andromeda was a separate galaxy to our own, numerous galaxies began to be surveyed as more powerful telescopes were built. This datasat aims to classificate stars, galaxies, and quasars based on their spectral characteristics.

## Background: how this data was collected
There are two different ways to measure light from an object in the sky:

- **Photometry (taking a picture):** A camera photographs the sky through a few colored filters and records how bright each object looks through each filter. This is quick and covers huge areas of sky, but gives only a few numbers per object.
- **Spectroscopy (splitting the light):** One object's light is spread out into a rainbow (a *spectrum*), like a prism does, and the brightness is measured at thousands of wavelengths. Chemical elements and motion leave specific "fingerprints" in that rainbow. This is much more detailed but slow and expensive, so it is only done for selected objects.

"Optical" means visible light (as opposed to radio, X-ray, etc.).

The SDSS works in two stages:

1. **Imaging stage:** A wide-field camera scans the sky and photographs it through five filters (u, g, r, i, z).
2. **Spectroscopy stage:** Some of the objects found in those images are chosen for spectroscopy. An aluminum **plate** is drilled with holes at the exact positions of those objects, and an optical **fiber** in each hole carries that object's light to a spectrograph. The spectrum is what astronomers use to decide the object's class and measure its redshift.

| Stage | Columns |
|---|---|
| Imaging | `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `u`, `g`, `r`, `i`, `z` |
| Spectroscopy | `spec_obj_ID`, `plate`, `fiber_ID`, `MJD`, `redshift`, `class` |

### What a spectrum is
A **spectrum** is an object's light sorted by color, from violet to red, with how much light there is at each color. Think of the rainbow you get when sunlight passes through a prism. Astronomers draw it as a graph of brightness against color (wavelength).

```
brightness
   │      ╱‾‾╲    ▲            ╱‾‾‾‾╲
   │  ╱‾‾╱    ╲__│__╱‾╲   ╱‾‾╱      ╲___
   │ ╱            │    ╲_╱ ▼
   └──────────────────────────────────────▶ color (wavelength)
     violet        blue   green    red
                  ▲ bright line   ▼ dark line
```

The graph isn't smooth. It has sharp **lines**:
- **Dark lines** are colors that are missing because atoms in the object absorbed them.
- **Bright lines** are colors that hot gas gives off.

Each chemical element always makes lines at its own set of colors, so the pattern of lines works like a **barcode**. Astronomers read this barcode to decide the class:
- **Star:** the barcode sits about where it would in a lab on Earth (redshift ≈ 0).
- **Galaxy:** the barcode is shifted toward red, because the galaxy is far away and moving away from us. **Redshift is the size of that shift.**
- **Quasar:** the barcode is shifted a long way, and it has very broad, bright lines from hot gas around a black hole.

**How this relates to u, g, r, i, z:** a spectrum measures thousands of colors, while the five filters measure only five broad bands. So the filter values are like a **five-pixel, blurry version of the spectrum**.

### How the imaging camera is laid out
```
        ◀───────── the sky drifts past this way as the Earth turns ─────────

 col 1  │ field 261 │ field 262 │ field 263 │ field 264 │ ...
 col 2  │ field 261 │ field 262 │ field 263 │ field 264 │ ...
 col 3  │ field 261 │ field 262 │ field 263 │ field 264 │ ...
 col 4  │ field 261 │ field 262 │ field 263 │ field 264 │ ...
 col 5  │ field 261 │ field 262 │ field 263 │ field 264 │ ...
 col 6  │ field 261 │ field 262 │ field 263 │ field 264 │ ...

        └──────────────── one run (one long sweep) ────────────────┘
```
- **Run:** The telescope holds still while the Earth turns, so the sky slides across the camera. One continuous sweep is one run.
- **Camera column (`cam_col`):** The camera has 6 columns of detectors side by side, and each one records its own long strip of sky.
- **Five filters per column:** Each column has five detectors in a line, one per filter (u, g, r, i, z). Every object passes over all five in turn, so every object gets all five brightness values.
- **Field:** Each strip is cut into consecutive rectangular images, numbered in order. Every column has its own field 263, so a field number on its own isn't unique. One image is identified by **`run_ID` + `rerun_ID` + `cam_col` + `field_ID`**, and you need all four.
- **Objects:** One field image contains hundreds to thousands of objects. Only a few of them are chosen for spectroscopy, and each of those spectra is one row in this dataset.

| SDSS | Analogy |
|---|---|
| Run (sweep) | A photographer walks down a street taking a long panorama |
| Camera column | The panorama is shot as 6 parallel strips |
| Field | Each strip is cut into photos. One photo shows a whole crowd. |
| Object | One person in that photo |
| Spectrum (one row) | A detailed medical scan of a few people chosen from the crowd |

## Content
The data consists of 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey). Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar.

1. `obj_ID` = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS.
    - *In simple terms:* An ID given to each object detected in the imaging stage. CAS (Catalog Archive Server) is the SDSS database.
2. `alpha` = Right Ascension angle (at J2000 epoch)
   - *In simple terms:* The object's east–west position on the sky, like longitude on a map. Ranges from 0° to 360°. "J2000" means the coordinates are fixed to the sky as it was in the year 2000, because Earth's axis slowly wobbles and the coordinate grid shifts over time.
3. `delta` = Declination angle (at J2000 epoch)
   - *In simple terms:* The object's north–south position on the sky, like latitude. Ranges from −90° (south pole of the sky) to +90° (north pole).
4. `u` = Ultraviolet filter in the photometric system
5. `g` = Green filter in the photometric system
6. `r` = Red filter in the photometric system
7. `i` = Near Infrared filter in the photometric system
8. `z` = Infrared filter in the photometric system
   - *In simple terms (u, g, r, i, z):* How bright the object looks through each of five colored filters, from ultraviolet (u) to infrared (z). The values are **magnitudes**, which work backwards: a **smaller number means brighter**. The scale is logarithmic, so a difference of 5 magnitudes means 100× brighter. The *difference* between two filters (e.g. `u − g`) is called a **color** and tells you whether the object gives off more blue or more red light.
9. `run_ID` = Run Number used to identify the specific scan
   - *In simple terms:* The camera photographs the sky by sweeping across a long strip of sky in one continuous session. Each sweep is a "run," and this is its number.
10. `rerun_ID` = Rerun Number to specify how the image was processed
    - *In simple terms:* The version of the image-processing software used to turn raw images into measurements. Like a software version number for the data.
11. `cam_col` = Camera column to identify the scanline within the run
    - *In simple terms:* The camera has six columns of detectors side by side (numbered 1–6), each covering its own parallel strip of sky during a run. This tells you which strip the object was in.
12. `field_ID` = Field number to identify each field
    - *In simple terms:* Each strip is cut into consecutive rectangular frames, called fields. This is the frame number. Together, `run_ID` + `rerun_ID` + `cam_col` + `field_ID` point to one specific image.
13. `spec_obj_ID` = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
    - *In simple terms:* An ID for one spectrum, meaning one measurement of an object's light split into a rainbow.
14. `class` = object class (galaxy, star or quasar object)
    - *In simple terms:*
      - **STAR:** A single star, almost always one inside our own galaxy, the Milky Way. Relatively close.
      - **GALAXY:** A huge system of billions of stars, gas and dust, far outside the Milky Way.
      - **QSO (quasar):** The extremely bright center of a very distant galaxy, powered by a supermassive black hole pulling in matter. Quasars are so far away that they look like a single point of light, like a star, in images.
15. `redshift` = redshift value based on the increase in wavelength
    - *In simple terms:* When an object moves away from us, its light gets stretched toward longer, redder wavelengths. Because the universe is expanding, more distant objects move away faster, so **larger redshift ≈ farther away**. Stars in our own galaxy have redshift near 0, and it can be slightly negative if the star is moving toward us. Redshift is measured from the spectrum.
16. `plate` = plate ID, identifies each plate in SDSS
    - *In simple terms:* The number of the drilled aluminum plate used for spectroscopy. Each plate has holes for several hundred objects in one patch of sky, all observed at the same time.
17. `MJD` = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
    - *In simple terms:* A date written as a single number: the count of days since 17 November 1858. Astronomers use it because subtracting two dates is easy. Here it is the night the spectrum was taken.
18. `fiber_ID` = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation
    - *In simple terms:* Which hole/fiber on the plate carried this object's light. `plate` + `MJD` + `fiber_ID` together point to one specific spectrum.

## Evaluation Metrics

### False Positive and False Negative

| Result Type | What It Means | Statistical Error Type | Common Example |
|---|---|---|---|
| **False Positive** | Test shows positive, but the condition is **not** there | Type I Error | A medical test shows disease when you are healthy; antivirus flags a safe file as a virus. |
| **False Negative** | Test shows negative, but the condition **is** there | Type II Error | A medical test misses an existing illness; security scanner misses hidden contraband. |


### Precision
- Definition: Precision measures how many of the model's positive predictions are actually correct.
- Formula:
$$
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$
- When to prioritize: Use precision when false positives are costly (e.g., flagging a legitimate email as spam).

### Recall
- Definition: Recall tracks the actual positive instances successfully found by the model.
- Formula:
$$
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
$$
- When to prioritize: Prioritize recall when false negatives are dangerous or costly, such as in medical diagnoses. 

### F1 Score
- Definition: The F1 score balances precision and recall as their harmonic mean, ranging from 0 to 1.
- Formula:
$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
- Why harmonic mean? It penalizes extreme imbalances between the two metrics.


## Acknowledgements
The data released by the SDSS is under public domain.
http://www.sdss.org/science/image-gallery/.
