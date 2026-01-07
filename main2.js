document.querySelector('.btn-create').addEventListener('click', async () => {
    const email = document.getElementById('email').value;
    const username = document.getElementById('Username').value;
    const password = document.getElementById('password').value;

    if (!email || !password || !username) return alert("Điền đủ thông tin!");

    const response = await fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, username, password })
    });

    const result = await response.json();
    alert(result.message);

    if (result.status === "success") {
        // Quay về trang chủ (Login)
        window.location.href = "/"; 
    }
});