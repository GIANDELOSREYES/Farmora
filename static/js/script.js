const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatContainer = document.getElementById("chat-container");


chatForm.addEventListener("submit", async function(event) {

    // Prevent the browser from refreshing the page
    event.preventDefault();

    const message = userInput.value.trim();

    // Prevent empty messages
    if (!message) {
        return;
    }

    // Display the user's message
    displayMessage(message, "user");

    // Clear the input
    userInput.value = "";

    // Display loading indicator
    const loadingMessage = displayMessage("Farmora is thinking...", "bot");

    try {

        // Send the question to Flask
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        // Convert Flask response into JavaScript object
        const data = await response.json();

        // Display the chatbot's answer
        loadingMessage.textContent = data.response;

    } catch (error) {

        console.error("Error:", error);

        loadingMessage.textContent =
            "Sorry, something went wrong. Please try again.";

    }

});


function displayMessage(message, sender) {

    const messageWrapper = document.createElement("div");

    messageWrapper.className =
        sender === "user"
        ? "bg-white p-4 rounded-2xl mt-4 w-fit max-w-[80%] justify-self-end mr-4"
        : "p-4 mt-4 w-fit max-w-[80%] ml-4";

    const messageText = document.createElement("p");

    messageText.className =
        "font-poppins text-base wrap-break-word";

    messageText.textContent = message;

    messageWrapper.appendChild(messageText);

    chatContainer.appendChild(messageWrapper);

    // Automatically scroll to the latest message
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return messageText;
}