//! Exercise the maintained Candle ZIP compatibility patch through the worker's
//! locked dependency graph. Based on huggingface/candle#4041.

use candle::npy::NpzTensors;
use candle::{Device, Error, Result, Tensor};
use std::borrow::Cow;
use std::io::{Read, Write};
use std::path::Path;

fn write_named_tensor(path: &Path, name: &str) -> Result<()> {
    let tensor = Tensor::new(&[2u32, 4, 6], &Device::Cpu)?;
    Tensor::write_npz(&[(name, &tensor)], path)?;
    Ok(())
}

fn assert_named_tensor(path: &Path, name: &str) -> Result<()> {
    let loaded = Tensor::read_npz(path)?;
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, name);
    assert_eq!(loaded[0].1.to_vec1::<u32>()?, vec![2, 4, 6]);
    let selected = Tensor::read_npz_by_name(path, &[name])?;
    assert_eq!(selected[0].to_vec1::<u32>()?, vec![2, 4, 6]);
    let lazy = NpzTensors::new(path)?;
    assert_eq!(
        lazy.names().iter().map(|n| n.as_str()).collect::<Vec<_>>(),
        vec![name]
    );
    let tensor = lazy.get(name)?.expect("the named tensor must be indexed");
    assert_eq!(tensor.to_vec1::<u32>()?, vec![2, 4, 6]);
    Ok(())
}

fn use_cp437_name(path: &Path, ascii_name: &[u8]) -> Result<()> {
    let mut bytes = std::fs::read(path)?;
    let positions: Vec<_> = bytes
        .windows(ascii_name.len())
        .enumerate()
        .filter_map(|(i, x)| (x == ascii_name).then_some(i))
        .collect();
    assert_eq!(positions.len(), 2);
    for index in positions {
        bytes[index] = 0x82;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn set_utf8_flag(path: &Path, enabled: bool) -> Result<()> {
    let mut bytes = std::fs::read(path)?;
    let mut headers = 0;
    for index in 0..bytes.len().saturating_sub(10) {
        let offset = match &bytes[index..index + 4] {
            b"PK\x03\x04" => 6,
            b"PK\x01\x02" => 8,
            _ => continue,
        };
        let mut flags = u16::from_le_bytes([bytes[index + offset], bytes[index + offset + 1]]);
        if enabled {
            flags |= 1 << 11;
        } else {
            flags &= !(1 << 11);
        }
        bytes[index + offset..index + offset + 2].copy_from_slice(&flags.to_le_bytes());
        headers += 1;
    }
    assert_eq!(headers, 2);
    std::fs::write(path, bytes)?;
    Ok(())
}

#[test]
fn npz_borrowed_utf8_names() -> Result<()> {
    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    write_named_tensor(&path, "données")?;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path)?)?;
    let entry = archive.by_index(0)?;
    assert!(matches!(entry.name()?, Cow::Borrowed("données.npy")));
    assert_named_tensor(&path, "données")
}

#[test]
fn npz_owned_cp437_names() -> Result<()> {
    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    write_named_tensor(&path, "a")?;
    use_cp437_name(&path, b"a.npy")?;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path)?)?;
    let entry = archive.by_index(0)?;
    assert!(matches!(entry.name()?, Cow::Owned(name) if name == "é.npy"));
    assert_named_tensor(&path, "é")
}

#[test]
fn npz_clear_utf8_flag_uses_zip9_utf8_first_decoding() -> Result<()> {
    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    write_named_tensor(&path, "é")?;
    set_utf8_flag(&path, false)?;
    // ZIP 7 returned "├⌐" for bytes C3 A9 with bit 11 clear; ZIP 9 uses UTF-8.
    assert_named_tensor(&path, "é")
}

#[test]
fn npz_set_utf8_flag_uses_zip9_cp437_fallback() -> Result<()> {
    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    write_named_tensor(&path, "a")?;
    use_cp437_name(&path, b"a.npy")?;
    set_utf8_flag(&path, true)?;
    // ZIP 7 returned U+FFFD for invalid UTF-8 byte 82; ZIP 9 falls back to CP437.
    assert_named_tensor(&path, "é")
}

#[test]
fn npz_unicode_path_overrides_raw_name() -> Result<()> {
    let source = tempfile::NamedTempFile::new()?.into_temp_path();
    write_named_tensor(&source, "a")?;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&source)?)?;
    let mut data = Vec::new();
    archive.by_index(0)?.read_to_end(&mut data)?;

    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    let mut archive = zip::ZipWriter::new(std::fs::File::create(&path)?);
    let mut options = zip::write::FullFileOptions::default();
    // Info-ZIP Unicode Path: version 1, CRC32 of b"a.npy", UTF-8 name.
    let mut unicode_path = vec![1];
    unicode_path.extend_from_slice(&0x663227a9u32.to_le_bytes());
    unicode_path.extend_from_slice("é.npy".as_bytes());
    options.add_extra_field(0x7075, unicode_path, false)?;
    archive.start_file("a.npy", options)?;
    archive.write_all(&data)?;
    archive.finish()?;

    assert_named_tensor(&path, "é")?;
    assert!(Tensor::read_npz_by_name(&path, &["a"]).is_err());
    Ok(())
}

#[test]
fn pth_utf8_and_cp437_names() -> Result<()> {
    for cp437 in [false, true] {
        let path = tempfile::NamedTempFile::new()?.into_temp_path();
        let mut archive = zip::ZipWriter::new(std::fs::File::create(&path)?);
        archive.start_file("a/data.pkl", zip::write::SimpleFileOptions::default())?;
        // torch.save({"weight": torch.tensor([2., 4., 6.])}), protocol 2.
        archive.write_all(
            &[
                b"\x80\x02}q\x00X\x06\x00\x00\x00weightq\x01ctorch._utils\n".as_slice(),
                b"_rebuild_tensor_v2\nq\x02((X\x07\x00\x00\x00storageq\x03",
                b"ctorch\nFloatStorage\nq\x04X\x01\x00\x00\x000q\x05",
                b"X\x03\x00\x00\x00cpuq\x06K\x03tq\x07QK\x00K\x03\x85q\x08",
                b"K\x01\x85q\x09\x89ccollections\nOrderedDict\nq\x0a)Rq\x0b",
                b"tq\x0cRq\x0ds.",
            ]
            .concat(),
        )?;
        archive.start_file("a/data/0", zip::write::SimpleFileOptions::default())?;
        for value in [2f32, 4., 6.] {
            archive.write_all(&value.to_le_bytes())?;
        }
        archive.finish()?;
        if cp437 {
            use_cp437_name(&path, b"a/data.pkl")?;
            use_cp437_name(&path, b"a/data/0")?;
        }
        let tensors = candle::pickle::read_all(&path)?;
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].0, "weight");
        assert_eq!(tensors[0].1.to_vec1::<f32>()?, vec![2., 4., 6.]);
    }
    Ok(())
}

#[test]
fn invalid_npz_returns_zip_errors() -> Result<()> {
    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    std::fs::write(&path, b"not a zip archive")?;
    assert!(matches!(Tensor::read_npz(&path), Err(Error::Zip(_))));
    assert!(matches!(NpzTensors::new(&path), Err(Error::Zip(_))));
    Ok(())
}

#[test]
fn invalid_pth_returns_zip_error() -> Result<()> {
    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    std::fs::write(&path, b"not a zip archive")?;
    assert!(matches!(
        candle::pickle::read_pth_tensor_info(&path, false, None),
        Err(Error::Zip(_))
    ));
    Ok(())
}

#[test]
fn selective_npz_preserves_unsupported_compression_error() -> Result<()> {
    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    write_named_tensor(&path, "a")?;
    let mut bytes = std::fs::read(&path)?;
    let mut headers = 0;
    for index in 0..bytes.len().saturating_sub(12) {
        let offset = match &bytes[index..index + 4] {
            b"PK\x03\x04" => 8,
            b"PK\x01\x02" => 10,
            _ => continue,
        };
        bytes[index + offset..index + offset + 2].copy_from_slice(&255u16.to_le_bytes());
        headers += 1;
    }
    assert_eq!(headers, 2);
    std::fs::write(&path, bytes)?;
    // The directory is readable; the error arises when opening its entry.
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path)?)?;
    assert!(matches!(
        archive.by_index(0),
        Err(zip::result::ZipError::CompressionMethodNotSupported(255))
    ));
    assert!(matches!(
        Tensor::read_npz_by_name(&path, &["a"]),
        Err(Error::Zip(
            zip::result::ZipError::CompressionMethodNotSupported(255)
        ))
    ));
    Ok(())
}

#[test]
fn selective_npz_reports_missing_array() -> Result<()> {
    let path = tempfile::NamedTempFile::new()?.into_temp_path();
    write_named_tensor(&path, "a")?;
    assert!(matches!(
        Tensor::read_npz_by_name(&path, &["missing"]),
        Err(Error::Npy(_))
    ));
    Ok(())
}
