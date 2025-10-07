import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { gzip } from 'zlib';
import { promisify } from 'util';

const gzipAsync = promisify(gzip);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Directory containing HTML files
const htmlDir = './public/skulls/';

// Get all HTML files
const files = fs.readdirSync(htmlDir).filter(file => file.endsWith('.html'));

console.log('Optimizing and compressing HTML files...\n');

for (const file of files) {
    const filePath = path.join(htmlDir, file);
    const originalSize = fs.statSync(filePath).size;
    
    console.log(`Processing ${file} (${(originalSize / 1024 / 1024).toFixed(2)} MB)...`);
    
    try {
        // Read the HTML file
        let content = fs.readFileSync(filePath, 'utf8');
        
        // Basic HTML optimization
        console.log('  - Removing extra whitespace...');
        content = content
            .replace(/\s+/g, ' ')  // Replace multiple whitespace with single space
            .replace(/>\s+</g, '><')  // Remove whitespace between tags
            .trim();
        
        // Write optimized version
        const optimizedPath = filePath.replace('.html', '_optimized.html');
        fs.writeFileSync(optimizedPath, content);
        
        const optimizedSize = fs.statSync(optimizedPath).size;
        const optimizationReduction = ((originalSize - optimizedSize) / originalSize * 100).toFixed(1);
        
        console.log(`  - Optimized to ${(optimizedSize / 1024 / 1024).toFixed(2)} MB (${optimizationReduction}% reduction)`);
        
        // Create compressed version of optimized file
        console.log('  - Creating compressed version...');
        const compressed = await gzipAsync(content);
        
        // Write compressed version
        const compressedPath = filePath.replace('.html', '_optimized.html.gz');
        fs.writeFileSync(compressedPath, compressed);
        
        const compressedSize = fs.statSync(compressedPath).size;
        const totalReduction = ((originalSize - compressedSize) / originalSize * 100).toFixed(1);
        
        console.log(`  ✓ Final compressed size: ${(compressedSize / 1024 / 1024).toFixed(2)} MB (${totalReduction}% total reduction)`);
        
    } catch (error) {
        console.log(`  ✗ Error processing ${file}:`, error.message);
    }
}

console.log('\nOptimization complete!');
console.log('\nFiles created:');
console.log('- *_optimized.html (optimized HTML)');
console.log('- *_optimized.html.gz (compressed optimized HTML)');
console.log('\nTo use the optimized files, update your IframeGrid to point to the _optimized.html files.');
