const { exec } = require('child_process');

function downloadVideo(youtubeUrl) {
    return new Promise((resolve, reject) => {
        const command = `youtube-dl ${youtubeUrl} -o "./videos/%(title)s.%(ext)s"`;
        exec(command, (error, stdout, stderr) => {
            if (error) {
                console.log(`error: ${error.message}`);
                reject(error);
                return;
            }
            if (stderr) {
                console.log(`stderr: ${stderr}`);
                reject(stderr);
                return;
            }
            console.log(`stdout: ${stdout}`);
            resolve(stdout);
        });
    });
}

module.exports = downloadVideo;
