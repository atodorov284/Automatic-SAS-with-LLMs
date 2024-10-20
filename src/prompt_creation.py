
class CreatePrompt:
    def __init__(self):
        self._questions = {1: 'A group of students wrote the following procedure for their investigation. Procedure:\
- Determine the mass of four different samples.\
- Pour vinegar in each of four separate, but identical, containers.\
- Place a sample of one material into one container and label. Repeat with remaining samples, placing a single sample into a single container.\
- After 24 hours, remove the samples from the containers and rinse each sample with distilled water.\
- Allow the samples to sit and dry for 30 minutes. \
- Determine the mass of each sample.\
The data recorded by the students are recorded in a table containing information about wood, plastic, marble and limestone and their respective starting mass, ending mass and difference in mass.\
After reading the group’s procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information.',

2 :'A student performed the following investigation to test four different polymer plastics for stretchability.\
Procedure:\
- Take a sample of one type of plastic, and measure its length.  \
- Tape the top edge of the plastic sample to a table so that it is hanging freely down the side of the table.\
- Attach a clamp to the bottom edge of the plastic sample.\
- Add weights to the clamp and allow them to hang for five minutes.\
- Remove the weights and clamp, and measure the length of the plastic types.\
- Repeat the procedure exactly for the remaining three plastic samples.\
- Perform a second trial (T2) exactly like the first trial (T1).\
The student recorded the data from the investigation in atable containing the stretches over the trials of the different kind of plastics.',

3 : 'Explain how pandas in China are similar to koalas in Australia and how they both are different from pythons. Support your response with information from the article.',

4 :'Explain the significance of the word “invasive” to the rest of the article. Support your response with information from the article.', 
5 :'Starting with mRNA leaving the nucleus, list and describe four major steps involved in protein synthesis.',
6 :'List and describe three processes used by cells to control the movement of substances across the cell membrane.',
7 :'Identify ONE trait that can describe Rose based on her conversations with Anna or Aunt Kolab. Include ONE detail from the story that supports your answer.',
8 :'During the story, the reader gets background information about Mr. Leonard. Explain the effect that background information has on Paul. Support your response with details from the story.',
9 :'How does the author organize the article? Support your response with details from the article.',
10 :'Brandi and Jerry did the following controlled experiment to find out how the color of an object affects its temperature.\
Question: What is the effect of different lid colors on the air temperature inside a glass jar exposed to a lamp?\
Hypothesis: The darker the lid color, the greater the increase in air temperature in the glass jar, because darker colors absorb more energy.\
Materials:\
glass jar\
lamp\
four colored lids: black, dark gray, light gray, and white\
thermometer\
meterstick\
stopwatch\
Procedure:\
Put the black lid with the attached thermometer on the glass jar.\
Make sure the starting temperature inside the jar is 24° C.\
Place lamp 5 centimeters away from the lid and turn on the lamp.\
After 10 minutes measure the air temperature inside the glass jar and record as Trial 1.\
Turn off lamp and wait until the air in the jar returns to the starting temperature.\
Repeat steps 2 through 5 for Trials 2 and 3.\
Repeat steps 1 through 6 for the dark gray, light gray, and white lids.\
Calculate and record the average air temperature for each lid color.\
Brandi and Jerry were designing a doghouse. Use the results from the experiment to describe the best paint color for the doghouse.\
In your description, be sure to:\
Choose a paint color. \
Describe how that color might affect the inside of the doghouse.\
Use results from the experiment to support your description.'
}

    def get_prompt(self, essay_set, final_score, readability, ttr, student_answer):
        questions = self._questions
        prompt = f'Pretend you are a teacher, having to grade students’ answers on a particular topic thoroughly, clearly, and objectively. This is the question being asked: {questions[essay_set]}. Considering that the final score obtained was {final_score}, the easy of readability scored at {readability}, and the text to token ratio was {ttr}, provide detailed feedback and key areas for improvement. This is the answer provided by the student: {student_answer}'
        return prompt