# test.py

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Function to test adding items to cart and checkout in a supermarket
def supermarket_test():
    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome()

    # Navigate to the supermarket website (replace with actual URL)
    driver.get("https://moreretail.in/supermarket")

    # Wait for the page to load (you may use explicit waits for real-world scenarios)
    time.sleep(3)

    # Search for an item (e.g., "Apples")
    search_box = driver.find_element(By.NAME, "search")
    search_box.send_keys("Apples")
    search_box.send_keys(Keys.RETURN)

    # Wait for search results to load
    time.sleep(2)

    # Add the first item in the search results to the cart (modify element locators as needed)
    add_to_cart_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Add to Cart')]")
    add_to_cart_button.click()

    # Wait for the item to be added to the cart
    time.sleep(2)

    # Proceed to checkout
    cart_button = driver.find_element(By.ID, "cart")
    cart_button.click()

    # Wait for cart page to load
    time.sleep(2)

    # Click the checkout button
    checkout_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Checkout')]")
    checkout_button.click()

    # Wait for checkout page to load
    time.sleep(2)

    # Fill out checkout form (replace with actual form field names/IDs)
    name_input = driver.find_element(By.ID, "name")
    address_input = driver.find_element(By.ID, "address")
    payment_input = driver.find_element(By.ID, "payment")

    name_input.send_keys("John Doe")
    address_input.send_keys("123 Market St, Springfield")
    payment_input.send_keys("4111111111111111")  # Fake credit card for testing

    # Confirm the purchase
    confirm_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Confirm')]")
    confirm_button.click()

    # Wait for confirmation message
    time.sleep(3)

    # Check if confirmation is displayed (you may use assertions here for real testing)
    confirmation_message = driver.find_element(By.ID, "confirmation").text
    print("Order Confirmation: ", confirmation_message)

    # Close the browser
    driver.quit()

if __name__ == "__main__":
    supermarket_test()