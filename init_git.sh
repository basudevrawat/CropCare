#!/bin/bash

# Initialize Git repository
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit of KisanSathi AI project"

# Instructions for GitHub
echo ""
echo "Repository initialized successfully!"
echo ""
echo "To push to GitHub, create a new repository on GitHub and then run:"
echo "git remote add origin https://github.com/yourusername/kisansathi-ai.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "Remember to replace 'yourusername' with your actual GitHub username." 