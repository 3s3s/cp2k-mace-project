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

// --- Абсолютные пути проекта ---
// __dirname  = .../src/webserver
// PROJECT_ROOT = .../cp2k-mace-project
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const RESULTS_DIR = path.join(PROJECT_ROOT, 'results');
const CONFIG_PATH = path.join(PROJECT_ROOT, 'src', 'config.json');
const MAIN_SCRIPT = path.join(PROJECT_ROOT, 'src', 'main.py');
const LOG_FILE = path.join(RESULTS_DIR, 'main.log');
const PID_FILE = path.join(RESULTS_DIR, 'main.pid');
const STATUS_FILE = path.join(RESULTS_DIR, 'main.status.json');
const ARCHIVE_FILE = path.join(RESULTS_DIR, 'artifacts.zip');

app.locals.paths = {
    PROJECT_ROOT, RESULTS_DIR, CONFIG_PATH, MAIN_SCRIPT,
    LOG_FILE, PID_FILE, STATUS_FILE, ARCHIVE_FILE,
};

// статика
app.use(express.static(path.join(__dirname, 'static_pages')));

// body parser для JSON
app.use(express.json({ limit: '2mb' }));

app.set('view engine', 'ejs');
// views по умолчанию в ./views
app.set('views', path.join(__dirname, 'views'));

app.use((err, req, res, next) => {
    console.error('Express error:', err);
    res.status(500).send('Something broke!');
});

require('./reqHandler.js').handle(app);