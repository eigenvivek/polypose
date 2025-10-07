import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Directory containing HTML files
const htmlDir = './public/skulls/';

// Get all HTML files
const files = fs.readdirSync(htmlDir).filter(file => file.endsWith('.html'));

console.log('Updating title display to be permanent...\n');

for (const file of files) {
    const filePath = path.join(htmlDir, file);
    
    console.log(`Processing ${file}...`);
    
    try {
        // Read the HTML file
        let content = fs.readFileSync(filePath, 'utf8');
        
        // Check if title functionality exists
        if (!content.includes('title-display')) {
            console.log(`  - No title functionality found in ${file}, skipping`);
            continue;
        }
        
        // Replace the existing title script with permanent version
        const permanentTitleScript = `
<script>
// Permanent title display functionality
(function() {
    // Get title from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const title = urlParams.get('title');
    
    if (title) {
        // Create permanent title overlay
        const titleOverlay = document.createElement('div');
        titleOverlay.id = 'title-display';
        titleOverlay.style.cssText = \`
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        \`;
        titleOverlay.textContent = title;
        
        // Add to body
        document.body.style.position = 'relative';
        document.body.appendChild(titleOverlay);
    }
})();
</script>`;
        
        // Find and replace the existing title script
        const scriptStart = content.indexOf('<script>\n// Title display functionality');
        const scriptEnd = content.indexOf('</script>', scriptStart) + 9;
        
        if (scriptStart !== -1 && scriptEnd !== -1) {
            content = content.slice(0, scriptStart) + permanentTitleScript + content.slice(scriptEnd);
            
            // Write the modified content back
            fs.writeFileSync(filePath, content);
            console.log(`  ✓ Updated ${file} to show permanent titles`);
        } else {
            console.log(`  ✗ Could not find existing title script in ${file}`);
        }
        
    } catch (error) {
        console.log(`  ✗ Error processing ${file}:`, error.message);
    }
}

console.log('\nTitle display update complete!');
console.log('Titles will now remain visible permanently within each iframe.');
