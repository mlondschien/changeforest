#[no_mangle]
pub extern "C" fn double_value(x: i32) -> i32 {
    println!("Just called a Rust function from C!");
    x * 2
}
