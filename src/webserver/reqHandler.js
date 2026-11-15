'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// ==== Управление дочерним процессом ====
let child = null; // ссылка на текущий запущенный main.py

function isAlive(pid) {
    if (!pid || !Number.isInteger(pid)) return false;
    try {
        process.kill(pid, 0);
        return true;
    } catch (e) {
        return false;
    }
}

function readPid() {
    try {
        const raw = fs.readFileSync(global.paths.PID_FILE, 'utf8').trim();
        const pid = parseInt(raw, 10);
        return Number.isInteger(pid) ? pid : null;
    } catch (e) {
        return null;
    }
}

function readStatus() {
    try {
        const raw = fs.readFileSync(global.paths.STATUS_FILE, 'utf8');
        return JSON.parse(raw);
    } catch (e) {
        return null;
    }
}

function logSize() {
    try {
        return fs.statSync(global.paths.LOG_FILE).size;
    } catch (e) {
        return 0;
    }
}

// ==================== Handlers ====================

exports.handle = function (app) {
    global.paths = app.locals.paths;

    app.get('/', onMain);
    app.get('/api/config', onGetConfig);
    app.post('/api/config', onSaveConfig);
    app.post('/api/run', onRun);
    app.post('/api/stop', onStop);
    app.get('/api/status', onStatus);
    app.get('/api/log', onLog);
    app.get('/download', onDownload);
};

// ---------- Главная страница ----------
function onMain(req, responce) {
    let render_info = { lang: 'ru' };
    responce.render('pages/index', render_info);
}

// ---------- Config ----------
function onGetConfig(req, res) {
    try {
        const raw = fs.readFileSync(global.paths.CONFIG_PATH, 'utf8');
        const parsed = JSON.parse(raw);
        res.json({ ok: true, config: parsed });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

function onSaveConfig(req, res) {
    try {
        const incoming = req.body && req.body.config;
        if (!incoming || typeof incoming !== 'object') {
            return res.status(400).json({ ok: false, error: 'config must be an object' });
        }

        // читаем текущий файл, чтобы сохранить help-тексты
        const raw = fs.readFileSync(global.paths.CONFIG_PATH, 'utf8');
        const current = JSON.parse(raw);

        const updated = {};
        for (const key of Object.keys(current)) {
            const cur = current[key];
            const inc = incoming[key];

            if (cur && typeof cur === 'object' && 'value' in cur && 'help' in cur) {
                // вариант {value, help}
                updated[key] = {
                    value: (inc && typeof inc === 'object' && 'value' in inc) ? inc.value : cur.value,
                    help: cur.help, // help не меняется
                };
            } else {
                // плоский ключ
                updated[key] = (inc !== undefined) ? inc : cur;
            }
        }

        // атомарная запись
        const tmp = global.paths.CONFIG_PATH + '.tmp';
        fs.writeFileSync(tmp, JSON.stringify(updated, null, 2), 'utf8');
        fs.renameSync(tmp, global.paths.CONFIG_PATH);

        res.json({ ok: true, config: updated });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

// ---------- Run / Stop ----------
function onRun(req, res) {
    try {
        // защита от двойного запуска
        const pid = readPid();
        if (pid && isAlive(pid)) {
            return res.status(409).json({ ok: false, error: 'already running', pid });
        }

        const args = [global.paths.MAIN_SCRIPT];
        const body = req.body || {};
        if (body.device === 'gpu') args.push('--gpu');
        else if (body.device === 'cpu') args.push('--cpu');

        // запускаем через python3, cwd = PROJECT_ROOT
        child = spawn('python3', args, {
            cwd: global.paths.PROJECT_ROOT,
            detached: false,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        // дублируем stdout/stderr в main.log на случай, если tee ещё не встал
        child.stdout.on('data', () => {});
        child.stderr.on('data', () => {});

        child.on('exit', (code, sig) => {
            console.log(`main.py exited code=${code} sig=${sig}`);
            child = null;
        });

        res.json({ ok: true, pid: child.pid });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

function onStop(req, res) {
    try {
        const pid = readPid() || (child && child.pid);
        if (!pid || !isAlive(pid)) {
            return res.status(404).json({ ok: false, error: 'not running' });
        }
        process.kill(pid, 'SIGTERM');
        res.json({ ok: true, pid });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

// ---------- Status ----------
function onStatus(req, res) {
    const pid = readPid();
    const alive = pid ? isAlive(pid) : false;
    const status = readStatus();
    res.json({
        ok: true,
        pid,
        alive,
        log_size: logSize(),
        status,
    });
}

// ---------- Log (incremental) ----------
function onLog(req, res) {
    try {
        const offset = parseInt(req.query.offset, 10);
        const from = Number.isFinite(offset) && offset >= 0 ? offset : 0;

        let data = '';
        let newOffset = from;
        try {
            const stat = fs.statSync(global.paths.LOG_FILE);
            if (from > stat.size) {
                // файл пересоздали — читаем с начала
                newOffset = 0;
            }
            const fd = fs.openSync(global.paths.LOG_FILE, 'r');
            const length = Math.max(0, stat.size - newOffset);
            if (length > 0) {
                const buf = Buffer.alloc(length);
                fs.readSync(fd, buf, 0, length, newOffset);
                data = buf.toString('utf8');
            }
            fs.closeSync(fd);
            newOffset = stat.size;
        } catch (e) {
            // файла нет — пусто
        }

        res.json({ ok: true, offset: newOffset, chunk: data });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

// ---------- Download ----------
function onDownload(req, res) {
    const file = global.paths.ARCHIVE_FILE;
    if (!fs.existsSync(file)) {
        return res.status(404).send('artifacts.zip not found yet');
    }
    res.download(file, 'artifacts.zip');
}