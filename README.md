![Muse](/resources/love_live_school_idol_project.jpg)

![status](https://img.shields.io/badge/status-work%20in%20progress-red)
![disclaimer](https://img.shields.io/badge/disclaimer-fun%20project-important)
# School Idol Training 
Yes I know, you have always dreamed of becoming an **idol**... But where to start ?

What if the solution was **Artificial Intelligence** ?

```Who said Artificial Intelligence ?! count me in !```  

![Kotori](/resources/kotori.jpeg)

This project provides anyone with a personal trainer. Just put a video of the choregraphy you want to learn and tada :tada: !

![Honoka](/resources/honoka.gif)

# Installation
Simply clone the project :
```bash
git clone https://github.com/Raffaelbdl/school-idol-training
cd school-idol-training
```
Install the requirements :
```bash
pip install -r requirements.txt
```
Install FFMPEG from their [website](https://www.ffmpeg.org/download.html)


Then run the module :
```bash
python -m sip
```

# Make your first choregraphy

```python
from sip import make_chore_from_file

chore = make_chore_from_file("$chore_title", "$path_to_video_file.mp4")
save_chore(chore, "./choregraphies/")
```

Then if you have chosen ```./choregraphies``` as your choregraphies folder, simply run the module :

```bash
python -m sip
```

# Resources
This project is powered by [Google Mediapipe API](https://google.github.io/mediapipe/).

# Plans for the future
Many improvements will be made in the future :
- [ ] Improving accuracy of the scoring system
- [ ] Giving score in real time
- [ ] Support for multi-person pose estimation
- [ ] Support for MMD choregraphies

# Disclaimer
This project was primarily made for fun :innocent:

There will be errors with scores, but high scores are most likely accurate if joint visibility is high
