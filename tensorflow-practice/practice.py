try:
    int(input("Enter a number: "))
except ValueError:
    print("Invalid input. Please enter a valid integer.")
else:
    print("Thank you for entering a valid number.")
finally:    
    print("This block will always execute, regardless of exceptions.")