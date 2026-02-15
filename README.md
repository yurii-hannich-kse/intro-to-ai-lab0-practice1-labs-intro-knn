# Lab 0 - Labs format, KNN (guided)

---

## What you do in Lab 0

During L0(P1) we will:
- Open and run a Jupyter notebook.
- Implement 3 small functions (they are automatically graded):
  - `euclidean_distance(point1, point2)`
  - `classify_example(X_train, y_train, test_point, k=3)`
  - `predict(X_train, y_train, X_test, k=3)`
- Compare your implementation with `sklearn`.

**More detailed instructions for lab is provided inside the notebook**

---

## Important rules

Follow these rules strictly:

- **Do not add any files to the repo**, only modify `lab.ipynb` (Github Classroom will automatically grade only this file and send alert if other files were changed).
- **Do not change the name of the notebook**.
- **Do not change the structure of the notebook** (e.g., move cells, add new ones, delete existing ones) - it may break the autograder. You can only modify the content of the cells that are already there and where it is clearly stated (like 'YOUR CODE HERE').
- **Document all the sources** used during work in `sources.txt`
- **Follow the AI Tools usage policy**

---

## Lab Workflow (simplified)

**This is simplified** step-by-step workflow for this and other labs. **For complete guide** refer to [**Lab Workflow Guide for Students**](https://teaching.kse.org.ua/pluginfile.php/135768/mod_resource/content/1/Lab%20Workflow%20Guide%20for%20Students.pdf) on the Moodle.


### 1. Open terminal and make sure you have git installed locally

```bash
git status
```

If an error occurs, download Git and Github CLI:
- **Git**: [Download here](https://git-scm.com/downloads)
- **GitHub CLI** (recommended): [Download here](https://cli.github.com/)

### 2. Authorize GitHub (Command Line)

To push/pull code from the command line, you need to authenticate once.
The GitHub CLI handles authentication automatically via your browser.

```bash
gh auth login
```

**Follow the prompts:**

1. **What account do you want to log into?** - `GitHub.com`
2. **What is your preferred protocol for Git operations?** - `HTTPS`
3. **Authenticate Git with your GitHub credentials?** - `Yes`
4. **How would you like to authenticate?** - `Login with a web browser`

A browser window opens:
- Log in with your **GitHub account**.
- Enter the **one-time code** shown in the terminal.
- Authorize the app.

**Verify:**

```bash
gh auth status
```

You should see: `Logged in to github.com as yourname`.

### 3. Clone Your Repository

Once you have logged in, navigate to where you want to store the lab:

```bash
cd ~/Desktop  # or any folder you prefer
```

Clone the repository:

```bash
git clone https://github.com/intro-ai-course/lab0-yourname.git
```

*Replace `lab0-yourname` with your actual repo name.*

Navigate into the folder:

```bash
cd lab0-yourname
```

### 4. Upload the working file to Google Colab

- Go to [colab.research.google.com](https://colab.research.google.com/).
- Click **File - Upload notebook**.
- Select `lab.ipynb` from your cloned repo folder.

### 5. Create `sources.txt` file locally

It will be your **submission file on the Moodle**.

Fill the file with this template
```
AI Chat: link_to_your_ai_chat
Web sources:
1. link_to_source_1
2. link_to_source_2
...
```

Create a new chat in selected AI tool (e.g. Gemini, ChatGPT or other) and paste the link to it into `sources.txt`. You need to fill it during the lab and submit to the Moodle.

There's no need to add this file to repository or push it. It will be analyzed separately by cource instructor or assistant, while the code of the lab will be autograded by Github Classroom.

### 6. Work on the Lab

#### 6.1 Working with code

Now fill out the lab notebook in Google Collab - edit code cells, run them, and test your implementation. 

**Important**: do not change the structure of the notebook (e.g., don't add/remove cells, don't change function signatures), otherwise the autograder may fail. Write the code only in the places that say `YOUR CODE HERE` or similar.

#### 6.2 Working with sources

When you google some information - paste the link to it into the `sources.txt`. 

All your conversation with AI must be in one chat that you've created on the previous step. **Use only allowed prompt templates** for your work:

**Prompt 1 - Understanding library components** 
> I am working with the [Library Name] library. Can you explain the internal logic and the main parameters of the [Function or Class Name]? Specifically, I want to understand what it does under the hood and what kind of data it expects as input and provides as the output. Do not write any code examples or pieces of solution.

**Prompt 2 - Requesting abstracted examples** 
> Can you provide a simple, abstracted code example of how to use [Function/Method Name]? Please use a dummy dataset like a small NumPy array or a basic list that is unrelated to Lab Topic, e.g., Iris dataset or Titanic data. I need to see the basic syntax for implementing this.

**Prompt 3 - Algorithmic logic explanation** 
> I need to implement the [Algorithm Name], e.g., KNN or Gradient Descent from scratch. Without providing any Python code, can you explain the step-by-step logic of how this algorithm processes data? Please use a first-principles approach to explain the mathematical or logical sequence.

**Prompt 4 - Debugging via traceback analysis** 
> I am getting an error in my code. I will provide the traceback error message below. Please analyze this error and explain where it is coming from and why it might be happening in the context of my logic. Important: Do not provide the corrected code - just help me understand the cause of the error so I can fix it myself.

Check the **AI Tools usage policy** of the course in the [Syllabus](https://teaching.kse.org.ua/mod/url/view.php?id=101969) or in the [Lab Workflow Guide for Students](https://teaching.kse.org.ua/pluginfile.php/135768/mod_resource/content/1/Lab%20Workflow%20Guide%20for%20Students.pdf) on the Moodle and follow it.

### 7. Saving the results

When the work is done, make sure all cells of the notebook run correctly (Runtime - Restart session and run all). 

If there are no errors and the result are correct - save the notebook:
- **File - Download - Download .ipynb**.
- Save it back into your **local cloned folder** (overwrite the original `lab.ipynb`).

Save the `sources.txt` as well with `Ctrl/Cmd + S`.

### 8. Commit and push the code

After editing `lab.ipynb`, you need to tell Git to track your changes.

#### Check Status

```bash
git status
```

You should see `lab.ipynb` listed as "modified".

#### Stage the File

```bash
git add lab.ipynb
```

#### Commit with a Message

```bash
git commit -m "Complete Lab 0"
```

*You can use any message, e.g., "Implement euclidean_distance", "Fix predict function", etc.*

#### Push to GitHub (Submit)

This uploads your work to GitHub and triggers the **autograder**.

```bash
git push origin main
```

### 9. Check Your Grade

1. Go to your repository on **GitHub.com**.
2. Click the **"Actions"** tab.
3. Click the latest workflow run (e.g., "GitHub Classroom Workflow").
4. You'll see test results:
   - **Green check** = Test passed.
   - **Red X** = Test failed.

You may also receive email notifications.

### 10. Submit `source.txt` on the Moodle

Go to assignment page on the Moodle and upload `source.txt` as file submission to the lab.

It will be additionally checked by course instructor or assistant on the subject of academic intergity. 

If you completed the lab and documented all the sources used - you'll get a lab grade that is defined by autograder. If there's something suspicios will be found (e.g. fully completed code without any sources and empty AI chat or forbidden prompts were used) it will require manual grading from course instructor or assistant.

---

**Good luck with your lab!**
