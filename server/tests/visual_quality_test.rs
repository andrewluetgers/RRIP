// Visual quality tests for tile generation
// Compares generated tiles against reference images to ensure no artifacts like dark banding

use anyhow::Result;
use image::{DynamicImage, GenericImageView, Pixel};
use std::path::Path;

/// Maximum allowed mean squared error (MSE) between pixels
const MAX_PIXEL_MSE: f64 = 100.0; // ~10 difference per channel

/// Maximum allowed percentage of pixels that can differ significantly
const MAX_DIFF_PERCENTAGE: f64 = 5.0;

/// Threshold for considering a pixel "significantly different"
const SIGNIFICANT_DIFF_THRESHOLD: u32 = 30; // Sum of absolute differences across channels

#[derive(Debug)]
struct ImageComparisonResult {
    mean_squared_error: f64,
    max_pixel_difference: u32,
    percent_pixels_different: f64,
    passed: bool,
}

/// Compare two images and return quality metrics
fn compare_images(img1: &DynamicImage, img2: &DynamicImage) -> Result<ImageComparisonResult> {
    let (width1, height1) = img1.dimensions();
    let (width2, height2) = img2.dimensions();

    if width1 != width2 || height1 != height2 {
        anyhow::bail!(
            "Image dimensions don't match: {}x{} vs {}x{}",
            width1, height1, width2, height2
        );
    }

    let mut total_squared_error = 0.0;
    let mut max_diff = 0u32;
    let mut significantly_different_pixels = 0;
    let total_pixels = (width1 * height1) as usize;

    let img1_rgb = img1.to_rgb8();
    let img2_rgb = img2.to_rgb8();

    for y in 0..height1 {
        for x in 0..width1 {
            let p1 = img1_rgb.get_pixel(x, y);
            let p2 = img2_rgb.get_pixel(x, y);

            let mut pixel_diff = 0u32;
            let mut pixel_squared_error = 0.0;

            for i in 0..3 {
                let diff = (p1[i] as i32 - p2[i] as i32).abs() as u32;
                pixel_diff += diff;
                pixel_squared_error += (diff * diff) as f64;
            }

            total_squared_error += pixel_squared_error;
            max_diff = max_diff.max(pixel_diff);

            if pixel_diff > SIGNIFICANT_DIFF_THRESHOLD {
                significantly_different_pixels += 1;
            }
        }
    }

    let mean_squared_error = total_squared_error / (total_pixels as f64 * 3.0);
    let percent_different = (significantly_different_pixels as f64 / total_pixels as f64) * 100.0;

    let passed = mean_squared_error <= MAX_PIXEL_MSE &&
                 percent_different <= MAX_DIFF_PERCENTAGE;

    Ok(ImageComparisonResult {
        mean_squared_error,
        max_pixel_difference: max_diff,
        percent_pixels_different: percent_different,
        passed,
    })
}

/// Detect dark banding artifacts in an image
fn detect_dark_banding(img: &DynamicImage) -> Result<bool> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Check for horizontal bands of very dark pixels
    let mut dark_band_rows = 0;

    for y in 0..height {
        let mut dark_pixels_in_row = 0;

        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            let brightness = (pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3;

            // Check if pixel is unexpectedly dark (< 10)
            if brightness < 10 {
                dark_pixels_in_row += 1;
            }
        }

        // If more than 80% of the row is dark, it's likely a banding artifact
        if dark_pixels_in_row as f64 > width as f64 * 0.8 {
            dark_band_rows += 1;
        }
    }

    // If we have multiple dark band rows, there's likely a problem
    Ok(dark_band_rows > 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_no_dark_banding_in_generated_tiles() -> Result<()> {
        // Start the server (or connect to running instance)
        let client = reqwest::Client::new();
        let base_url = "http://localhost:8080";

        // Wait for server to be ready
        for _ in 0..10 {
            if client.get(format!("{}/healthz", base_url))
                .send()
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Test a few L0 tiles that should be reconstructed
        // These coordinates correspond to the L2 tile at (4, 4) in level 14
        let test_tiles = vec![
            ("demo_out", 15, "16_16"),  // L1 tile from L2 (8,8)
            ("demo_out", 15, "17_16"),  // L1 tile
            ("demo_out", 15, "16_17"),  // L1 tile
        ];

        for (slide_id, level, coord) in test_tiles {
            println!("Testing tile {}/{}/{}.jpg", slide_id, level, coord);

            let url = format!("{}/tiles/{}/{}/{}.jpg", base_url, slide_id, level, coord);
            let response = timeout(Duration::from_secs(5), client.get(&url).send())
                .await??;

            assert_eq!(response.status(), 200, "Failed to fetch tile");

            let bytes = response.bytes().await?;
            let img = image::load_from_memory(&bytes)?;

            // Check for dark banding
            let has_banding = detect_dark_banding(&img)?;
            assert!(!has_banding,
                "Dark banding detected in tile {}/{}/{}.jpg",
                slide_id, level, coord);

            // Additional sanity checks
            let (width, height) = img.dimensions();
            assert_eq!(width, 256, "Tile width should be 256");
            assert_eq!(height, 256, "Tile height should be 256");

            // Check that the image isn't mostly black
            let rgb = img.to_rgb8();
            let mut total_brightness = 0u64;
            for pixel in rgb.pixels() {
                total_brightness += pixel[0] as u64 + pixel[1] as u64 + pixel[2] as u64;
            }
            let avg_brightness = total_brightness / (width as u64 * height as u64 * 3);

            assert!(avg_brightness > 20,
                "Image is too dark (avg brightness: {}), possible generation issue",
                avg_brightness);
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Run with --ignored when reference images are available
    async fn test_tile_quality_vs_reference() -> Result<()> {
        let client = reqwest::Client::new();
        let base_url = "http://localhost:8080";

        // Wait for server
        for _ in 0..10 {
            if client.get(format!("{}/healthz", base_url))
                .send()
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Compare generated tiles with reference images
        let reference_dir = Path::new("test-data/reference_tiles");
        if !reference_dir.exists() {
            eprintln!("Skipping reference comparison - no reference tiles found at {:?}", reference_dir);
            return Ok(());
        }

        for entry in fs::read_dir(reference_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("jpg") {
                let filename = path.file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| anyhow::anyhow!("Invalid filename"))?;

                // Parse filename format: {slide_id}_{level}_{x}_{y}.jpg
                let parts: Vec<&str> = filename.split('_').collect();
                if parts.len() != 4 {
                    continue;
                }

                let slide_id = parts[0];
                let level = parts[1];
                let coord = format!("{}_{}", parts[2], parts[3]);

                println!("Comparing {}/{}/{}.jpg with reference", slide_id, level, coord);

                // Load reference image
                let reference_img = image::open(&path)?;

                // Fetch generated tile
                let url = format!("{}/tiles/{}/{}/{}.jpg", base_url, slide_id, level, coord);
                let response = client.get(&url).send().await?;
                assert_eq!(response.status(), 200);

                let bytes = response.bytes().await?;
                let generated_img = image::load_from_memory(&bytes)?;

                // Compare images
                let comparison = compare_images(&reference_img, &generated_img)?;

                println!("  MSE: {:.2}, Max diff: {}, Different pixels: {:.2}%",
                    comparison.mean_squared_error,
                    comparison.max_pixel_difference,
                    comparison.percent_pixels_different);

                assert!(comparison.passed,
                    "Quality check failed for {}/{}/{}.jpg: {:?}",
                    slide_id, level, coord, comparison);
            }
        }

        Ok(())
    }

    #[test]
    fn test_image_comparison_logic() {
        // Create two identical test images
        let img1 = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            100, 100,
            image::Rgb([128, 128, 128])
        ));

        let img2 = img1.clone();

        let result = compare_images(&img1, &img2).unwrap();
        assert_eq!(result.mean_squared_error, 0.0);
        assert_eq!(result.max_pixel_difference, 0);
        assert_eq!(result.percent_pixels_different, 0.0);
        assert!(result.passed);

        // Create an image with slight differences
        let mut img3 = img1.to_rgb8();
        for x in 0..10 {
            for y in 0..10 {
                img3.put_pixel(x, y, image::Rgb([138, 138, 138]));
            }
        }

        let result = compare_images(&img1, &DynamicImage::ImageRgb8(img3)).unwrap();
        assert!(result.mean_squared_error > 0.0);
        assert_eq!(result.max_pixel_difference, 30); // 10 diff per channel
        // 100 pixels changed out of 10000, but diff is exactly at threshold (30)
        // so percent_pixels_different is 0
        assert_eq!(result.percent_pixels_different, 0.0);

        // Create image with more significant differences
        let mut img4 = img1.to_rgb8();
        for x in 0..10 {
            for y in 0..10 {
                img4.put_pixel(x, y, image::Rgb([160, 160, 160])); // 32 diff per channel
            }
        }

        let result = compare_images(&img1, &DynamicImage::ImageRgb8(img4)).unwrap();
        assert!(result.mean_squared_error > 0.0);
        assert_eq!(result.max_pixel_difference, 96); // 32 * 3 channels
        assert!(result.percent_pixels_different > 0.0); // Should be 1% (100/10000)
    }

    #[test]
    fn test_dark_banding_detection() {
        // Create image with dark bands
        let mut img = image::RgbImage::new(256, 256);

        // Fill with normal gray
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([128, 128, 128]);
        }

        // Add dark bands
        for y in 50..60 {
            for x in 0..256 {
                img.put_pixel(x, y, image::Rgb([5, 5, 5]));
            }
        }

        for y in 100..110 {
            for x in 0..256 {
                img.put_pixel(x, y, image::Rgb([3, 3, 3]));
            }
        }

        let dynamic_img = DynamicImage::ImageRgb8(img);
        let has_banding = detect_dark_banding(&dynamic_img).unwrap();
        assert!(has_banding, "Should detect dark banding");

        // Create normal image without banding
        let normal_img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            256, 256,
            image::Rgb([128, 128, 128])
        ));

        let has_banding = detect_dark_banding(&normal_img).unwrap();
        assert!(!has_banding, "Should not detect banding in normal image");
    }
}