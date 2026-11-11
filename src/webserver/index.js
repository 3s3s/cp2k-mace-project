'use strict';

const path = require('path');
const g_constants = require('./constants');

const https = require('https');
const express = require('express');

const app = express();

const httpsServer = https.createServer(g_constants.SSL_options, app);

const httpsListener = httpsServer.listen(g_constants.SSL_PORT, () => {
    console.log("SSL Proxy listening on port " + g_constants.SSL_PORT);
});

httpsListener.on('connection', (socket) => {
    socket.on('close', () => {});
});

// статика — абсолютный путь относительно index.js
app.use(express.static(path.join(__dirname, 'static_pages')));
app.set('view engine', 'ejs');

app.use((err, req, res, next) => {
    res.status(500).send('Something broke!');
});

require('./reqHandler.js').handle(app);