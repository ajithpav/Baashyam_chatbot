function createChatBot(host, title, welcomeMessage, inactiveMsg) {
    // Set initial welcome message
    document.querySelector('.chat-area').innerHTML = `<div class="bot-msg"><div class="msg">${welcomeMessage}</div></div>`;

    document.querySelector('.chat-submit').addEventListener('click', function() {
        let userMessage = document.querySelector('.chat-input').value;

        if (userMessage.trim()) {
            appendUserMessage(userMessage);
            document.querySelector('.chat-input').value = '';

            // Send message to the backend
            fetch(host, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage })
            })
            .then(response => response.json())  // Get the JSON response
            .then(data => {
                appendBotMessage(data.response);  // Access only the 'response' part of the JSON
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    });

    // Function to append user messages
    function appendUserMessage(message) {
        const chatArea = document.querySelector('.chat-area');
        let userMsgDiv = document.createElement('div');
        userMsgDiv.classList.add('user-msg');
        userMsgDiv.innerHTML = `<div class="msg">${message}</div>`;
        chatArea.appendChild(userMsgDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // Function to append bot messages
    function appendBotMessage(message) {
        const chatArea = document.querySelector('.chat-area');
        let botMsgDiv = document.createElement('div');
        botMsgDiv.classList.add('bot-msg');
        botMsgDiv.innerHTML = `<div class="msg">${message}</div>`;
        chatArea.appendChild(botMsgDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // Toggle Chatbot visibility and hide icon
    document.getElementById('chatbot-icon').addEventListener('click', function() {function createChatBot(host, title, welcomeMessage, inactiveMsg) {
        // Set initial welcome message
        document.querySelector('.chat-area').innerHTML = `<div class="bot-msg"><div class="msg">${welcomeMessage}</div></div>`;
    
        document.querySelector('.chat-submit').addEventListener('click', function() {
            let userMessage = document.querySelector('.chat-input').value;
    
            if (userMessage.trim()) {
                appendUserMessage(userMessage);
                document.querySelector('.chat-input').value = '';
    
                // Send message to the backend
                fetch(host, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: userMessage })
                })
                .then(response => response.json())  // Parse the JSON response
                .then(data => {
                    if (data.response) {
                        // Ensure that only the value inside 'response' is appended
                        appendBotMessage(data.response);  // Access the response text
                    } else {
                        appendBotMessage("Sorry, something went wrong.");
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    appendBotMessage("Error: Unable to connect to the server.");
                });
            }
        });
    
        function appendUserMessage(message) {
            const chatArea = document.querySelector('.chat-area');
            let userMsgDiv = document.createElement('div');
            userMsgDiv.classList.add('user-msg');
            userMsgDiv.innerHTML = `<div class="msg">${message}</div>`;
            chatArea.appendChild(userMsgDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    
        function appendBotMessage(message) {
            const chatArea = document.querySelector('.chat-area');
            let botMsgDiv = document.createElement('div');
            botMsgDiv.classList.add('bot-msg');
            botMsgDiv.innerHTML = `<div class="msg">${message}</div>`;
            chatArea.appendChild(botMsgDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    
        document.getElementById('chatbot-icon').addEventListener('click', function() {
            document.getElementById('chatbot-ui').classList.add('show');
            document.getElementById('chatbot-icon').classList.add('hide');
        });
    
        document.querySelector('.chat-header button').addEventListener('click', function() {
            document.getElementById('chatbot-ui').classList.remove('show');
            document.getElementById('chatbot-icon').classList.remove('hide');
        });
    }
    
        document.getElementById('chatbot-ui').classList.add('show');
        document.getElementById('chatbot-icon').classList.add('hide');
    });

    // Close chatbot when clicking "X"
    document.querySelector('.chat-header button').addEventListener('click', function() {
        document.getElementById('chatbot-ui').classList.remove('show');
        document.getElementById('chatbot-icon').classList.remove('hide');
    });
}

function handleButtonClick(message) {
    const userMessage = `<div class="user-msg"><div class="msg">${message}</div></div>`;
    document.querySelector('.chat-area').innerHTML += userMessage;
    // Send the button click message to your backend for processing
    fetch('http://localhost:5005/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        const botMessage = `<div class="bot-msg"><div class="msg">${data.response}</div></div>`;
        document.querySelector('.chat-area').innerHTML += botMessage;
    })
    .catch(error => console.error('Error:', error));
}
