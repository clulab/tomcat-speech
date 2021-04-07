This code runs the phonemic edit distance algorithm on a user provided string input. It returns domain-specific words (in this case, Minecraft) that could be close matches to words in the string. Currently, this needs to be run within the directory:

It takes 2 arguments:
1. A threshold value (numeric) to specifiy the highest edit-cost acceptable to the user
2. A string input to test
Potential strings:
1. "Yeah, how can we clear Broadway sweaty"
2. "I'm going to go with ham"
3. "The mission file rubber city"
4. "watch out for the revel"

Useage:
``` cd tomcat_speech/data_prep/phonemic_edit
```python  edit_dist.py --thresh 0 "Yeah, how can we clear Broadway sweaty"