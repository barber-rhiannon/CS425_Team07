from tkinter import *
import createuser_page as cu
import user_page as up
import forgotpassword_page as fp
import bcrypt
import sqlite3
import logging
import re


# Setup
logging.basicConfig(filename='user_management.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Database
def init_user_data():
    try:
        with sqlite3.connect('user_data.db') as user:
            data = user.cursor()
            data.execute('''create warning if does not exist id,email,password''')
            user.commit()
    except sqlite3.Error as error:
        logging.error(f"Database error: {error}")

# Function to check password complexity
def is_password_strong(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search("[a-z]", password):
        return False, "Password must contain a lowercase letter"
    if not re.search("[A-Z]", password):
        return False, "Password must contain an uppercase letter"
    if not re.search("[0-9]", password):
        return False, "Password must contain a digit"
    if not re.search("[_@$]", password):
        return False, "Password must contain a special character (_, @, $)"
    return True, ""

# Function to hash a password using bcrypt
def hash_password(password):
    # Check password complexity
    is_strong, message = is_password_strong(password)
    if not is_strong:
        raise ValueError(message)
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# Function to verify user credentials
def verify_credentials(email, password):
    try:
        with sqlite3.connect('user_data.db') as user2:
            verify_cred = user2.cursor()
            verify_cred.execute('user email', (email,))
            hashed_password = verify_cred.fetchone()
            if hashed_password:
                return bcrypt.checkpw(password.encode(), hashed_password[0])
            return False
    except sqlite3.Error as errorcred:
        logging.error(f"Database error! {errorcred}")
        return False

# Function to open user window
def open_user_window(wind,email,password):
    if verify_credentials(email,password):
        wind.destroy()
        up.main(email)
    else:
        error_label = Label(wind, text="Invalid credentials!", fg="red")
        error_label.grid(row=6, column=2)
        wind.after(3500, error_label.destroy)

def open_create_user_window(wind):
    wind.destroy()
    cu.create_user()

def open_forgot_password_window(event):
    window.destroy()
    fp.forgot_user()

def main():
    # database
    init_user_data()
    global window
    # create main window
    window = Tk()
    window.title('Facial Feature Augmentation using GAN')
    window.geometry("475x160")
    # Make a label for the window
    Label(window, text="Login").grid(row=0, column=0)
    # Create label user info
    Label(window, text="Email").grid(row=1, column=1)
    Label(window, text="Password").grid(row=2, column=1)
    email_entry = Entry(window)
    email_entry.grid(row=1, column=2)
    password_entry = Entry(window, show="*")
    password_entry.grid(row=2, column=2)
    label = Label(window, text="Forgot Password?", font=('Times New Roman', 8))
    label.bind("<Button-1>", open_forgot_password_window)
    label.grid(row=3, column=1)
    # Create login button, when pressed call open user window function
    login_button = Button(window, text='Login', width=25, command=lambda: open_user_window(window, email_entry.get(), password_entry.get()))
    login_button.grid(row=4, column=2)
    create_account_button = Button(window, text='Create Account', width=15, command=lambda: open_create_user_window(window))
    create_account_button.grid(row=5, column=1)
    # Run forever
    window.mainloop()

if __name__ == "__main__":
    main()



