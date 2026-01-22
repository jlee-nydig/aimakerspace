# 🧑‍💻 Your Version Control Workflow

We can break down your specific workflow into two distinct Gitflows for the course:

1. Initial Setup: Setting Up Your Local and Remote Git Repos
2. Weekly Workflow: Working on Assignments

In this walkthrough, we’ll cover the Weekly Workflow

## **Part I: Setting Up Your Local and Remote Git Repos**

You might be wondering: "*How do I make changes to this very repo that I’m reading right now?*" 

Short answer: **You don’t**!

Let’s set up the repo that you *will make changes to every week when doing your assignments.*

#### 0️⃣ Pre-Requisites
First, do these things:

- [Set up your SSH key on GitHub.com](https://github.com/AI-Maker-Space/Interactive-Dev-Environment-for-AI-Engineers?tab=readme-ov-file#-setting-up-keys-and-tokens).
- If you're on Windows, [set up Windows Subsystem for Linux (WSL2)](https://github.com/AI-Maker-Space/Interactive-Dev-Environment-for-AI-Engineers?tab=readme-ov-file#rocket-lets-get-started).

####  1️⃣ Create a Brand New GitHub Repo
You can follow [this guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories) if you need to, but creating a blank repo is pretty straightforward. A few notes on the seeing you should use, as shown below.

1. Please use the **repository name** that corresponds to your cohort! e.g., AIE, etc. Don't get fancy with the name - your life will be easier if you keep the name simple.
2. The process will be easiest if you make sure **Add a README file is deselected**.

> ✅ **Checkpoint**
> You’ve just created an empty remote repository on GitHub.

####  2️⃣ Get [Secure Shell Protocol (SSH)](https://en.wikipedia.org/wiki/Secure_Shell) Address
Once you've created your new repository, copy the repo’s SSH address. You'll want to copy this address as shown below. Click the 'copy' icon at the end of the address bar 

![image](https://i.imgur.com/62QNyfz.png)

####  3️⃣ Clone Your GitHub Repo Locally
Execute the command:

```
git clone git@github.com:yourusername/yourrepo.git
```

Then we need to Change Directory, or `cd` into our newly cloned repository!

```
cd yourrepo
```

> *If you see a warning message like: `warning: You appear to have cloned an empty repository`that means you've done everything right!
> 

*You’ve just cloned your empty remote repo locally to your machine using an SSH key.*

####  4️⃣ Add Class Repo as an Upstream Remote

First, run this command (but make sure to replace the XX with your cohort number):

```markdown
git remote add upstream git@github.com:AI-Maker-Space/AIE.git
```

Verify both remotes are connected. *You should see both "origin" (your repo) and "upstream" (class repo)*.

```
git remote -v
```

You should see an output very similar to this (your origin will be a different address)

```
origin  git@github.com:chris-alexiuk/AIE.git (fetch)
origin  git@github.com:chris-alexiuk/AIE.git (push)
upstream        git@github.com:AI-Maker-Space/AIE.git (fetch)
upstream        git@github.com:AI-Maker-Space/AIE.git (push)

```

> ✅ **Checkpoint**
> You’ve just connected AI Makerspace’s remote class repo (which we manage) to your remote class repo (which you manage).

#### 5️⃣ Your First [Pull](https://git-scm.com/docs/git-pull)
It’s time to pull down the course materials from AI Makerspace’s remote repository, which is upstream of both your local and remote repos.

> *You will do this each week as new assignments are released!
> 

```markdown
git pull upstream main --allow-unrelated-histories
```

> ✅ **Checkpoint**
> You’ve just pulled down course materials from AI Makerspace’s remote repo, **down to your local repository**.

#### 6️⃣ Your First [Push](https://git-scm.com/docs/git-push)
Now we need to push everything up from our local repo to our remote repository on GitHub.com

```markdown
git push origin main
```
> ✅ **Checkpoint**
> You’ve just pushed the course materials in your local repo up to your remote repo (origin) on GitHub.com. 

> During steps 5️⃣ and 6️⃣, notice the use of `main`.`main` is simply the name given to the [default branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches#about-the-default-branch) in a repo.
> 

## **Part II: Working on Assignments**

You might be wondering: "*How do I actually do my assignments?* 

Short answer: **Like this!**

![image](https://i.imgur.com/7TA9TIu.png)

Let’s walk through the process that you’ll use to *work on assignments and submit them via your remote repository on GitHub.com*.

#### 0️⃣ Pre-Requisites 
First, make sure that you’ve followed `Part I: Setting Up Your Local and Remote Git Repos` 

Now, imagine you want to pull down week 2’s work *as the assignment is released as class begins*!

####  1️⃣ Pull New Course Materials
Run this command:

```markdown
git pull upstream main --allow-unrelated-histories
```

#### 2️⃣ Do Assignment
Make changes, do the homework

`... do your work ...`

Run this command before to prepare your new (staged) content for a commit (to production, let’s say).

```markdown
git add .
```

> ✅ **Checkpoint**
> You just moved changes from your working directly to the staging area (also called the index).

#### 3️⃣ Your First [Commit](https://git-scm.com/docs/git-commit)
Create a new commit that includes a log message describing the changes you’ve made.

```markdown
git commit -m "Completed lesson 1 assignment"
```

*You just recorded the changes to your repo.*

#### 4️⃣ Push to Your Remote
Now that we’re done with our assignment, we can overwrite the old unfinished assignment on our remote repository with our new finished code!

```markdown
git push origin main
```

> ✅ **Checkpoint**
> You just pushed your changes to production (your live, remote, always-on repo).

#### 5️⃣ Repeat Weekly!
Now imagine you’re about to start a new assignment! Can you recall the steps you need to follow and why?

```markdown
git pull upstream main --allow-unrelated-histories # Get new lesson materials from AI Makerspace remote
#--do work--
git add . # Add changes to git history / move changes to staging 
git commit -m "Completed lesson X assignment" # Commit changes to git log with a helpful message
git push origin main # Push changes to our public remote so we can submit!
```

### Thinking Questions

- Can you look at the diagram with confidence?
- What are three lessons you've learned from this?
- What is one [lesson that you have not yet learned](https://www.loom.com/share/b34e4bd657f74892ac9a01f774113b4d)?
