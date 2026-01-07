document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    const response = await fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });

    const result = await response.json();
    alert(result.message);
    if (result.status === "success") {
    // Flask sẽ nhận yêu cầu này và chạy hàm selection() trả về index4.html
    window.location.href = "/selection"; 
}
    else {
        // Nếu lỗi, mới hiện thông báo lỗi từ server gửi về
        alert(result.message || "Đã có lỗi xảy ra");
    }
});

// Chuyển sang trang tạo tài khoản khi bấm nút Create Account
document.querySelector('.btn-create').addEventListener('click', () => {
    window.location.href = "/register-page"; 
});

// Chuyển sang giao diện tạo tài khoản
document.querySelector('.btn-create').addEventListener('click', () => {
    window.location.href = "/register-page"; 
});

