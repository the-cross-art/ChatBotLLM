#!/usr/bin/env node

const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/ws');

ws.on('open', function open() {
    console.log('✅ Connected to WebSocket');

    // Send init message
    ws.send(JSON.stringify({
        user_id: 'test_user',
        thread_id: 'test_thread',
        init: true
    }));
});

ws.on('message', function message(data) {
    console.log('📨 Received:', data.toString());

    // After successful init, send a test message
    if (data.toString().includes('initialized')) {
        console.log('✅ Sending test message...');
        ws.send(JSON.stringify({
            message: 'Hello, this is a test message!'
        }));
    }
});

ws.on('close', function close() {
    console.log('❌ Connection closed');
});

ws.on('error', function error(err) {
    console.error('🚨 WebSocket error:', err);
});

// Close connection after 30 seconds
setTimeout(() => {
    console.log('⏰ Closing connection...');
    ws.close();
}, 30000);