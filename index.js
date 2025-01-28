const express = require('express');
const bodyParser = require('body-parser');
const translate = require('google-translate-api-x');
const fetch = require('node-fetch');
const cors = require("cors");

const app = express();
const PORT = 3002;

app.use(bodyParser.json());
app.use(cors());
async function suggestWords(character) {
    const response = await fetch(`https://api.datamuse.com/words?sp=${character}*`);
    const data = await response.json();
    return data.map(word => word.word);
}

app.post('/suggest', async (req, res) => {
    const { character } = req.body;
    if (!character || typeof character !== 'string') {
        return res.status(400).json({ error: 'Invalid character input' });
    }

    try {
        let suggestions = await suggestWords(character);
        suggestions.sort((a, b) => a.length - b.length);
        const topSuggestions = suggestions.slice(0, 5);
        res.json(topSuggestions);
    } catch (error) {
        res.status(500).json({ error: 'Error fetching suggestions' });
    }
});


app.post('/translate', async (req, res) => {
    const { text, language } = req.body;
    if (!text || !language) {
        return res.status(400).json({ error: 'Invalid input. Both text and language are required.' });
    }

    try {
        const result = await translate(text, { to: language });
        res.status(200).json({ translatedText: result.text });
    } catch (err) {
        res.status(500).json({ error: 'Translation failed', details: err.message });
    }
});


app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
