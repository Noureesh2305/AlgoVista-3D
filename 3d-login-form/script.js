const title = document.getElementById("title");
const emailField = document.getElementById("emailField");
const loginBtn = document.getElementById("loginBtn");
const signupBtn = document.getElementById("signupBtn");

signupBtn.onclick = function () {
  title.innerText = "Register";
  emailField.style.display = "block";
  loginBtn.innerText = "Register";
};

loginBtn.onclick = function (e) {
  if (title.innerText === "Register") {
    alert("Registration Successful!");
    e.preventDefault();
  }
};
