

# Set your global Git identity
git config --global user.name "Keira Robinson"
git config --global user.email "krobinson11@gmail.com"

# Verify configuration
git config --global --list



# Stage and commit your changes
git add .
git commit -m "Renamed train_model.py and updated files"

# Pull remote changes and rebase
git pull origin main --rebase

# Push after successful rebase
git push origin main
