'use strict';

const fs = require('fs');
const path = require('path');

exports.SSL_PORT = 8443;

exports.SSL_options = {
    key:  fs.readFileSync(path.join(__dirname, 'privkey.pem')),
    cert: fs.readFileSync(path.join(__dirname, 'fullchain.pem')),
};